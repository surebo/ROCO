import copy

import numpy as np

from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop
from ER.PER.prioritized_memory import PER_Memory
import torch.nn as nn


# just start
class Esip:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents
        self.device = args.device
        self.params = list(mac.parameters())
        self.lam = args.lam
        self.alpha = args.alpha
        self.ind = args.ind
        self.mix = args.mix
        self.expl = args.expl
        self.dis = args.dis
        self.goal = args.goal
        self.device = args.device
        
        

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.target_mixer = copy.deepcopy(self.mixer)

        self.mixer_params = list(self.mixer.parameters())
        self.q_params = list(mac.parameters())
        self.mixer_optimiser = RMSprop(params=self.mixer_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.q_optimiser = RMSprop(params=self.q_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.distance = nn.Sequential(
            nn.Linear(self.mac.scheme1['obs']['vshape'], 128),
            nn.ReLU(),
            nn.Linear(128, args.n_actions)
        ).to(device=self.device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        reward = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        q_errors = batch["q_error"][:, :-1]      
        observation = batch["obs"][:, :-1]         
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        indi_terminated = batch["indi_terminated"][:, :-1].float()

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time    根据obs选择 动作 实际动作

        # Pick the Q-Values for the actions taken by each agent

        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        ind_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets

        target_ind_q = th.stack(target_mac_out[:-1], dim=1)  #### For Q value s
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999
        target_ind_q[avail_actions[:, :-1] == 0] = -9999999  # Q values  ##########################

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            

            cur_max_act = mac_out_detach[:, :-1].max(dim=3, keepdim=True)[1]  ##########################
            

            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            target_individual_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            target_ind_qvals = th.gather(target_ind_q, 3, cur_max_act).squeeze(3)  ######################
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]
            target_individual_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals_clone = chosen_action_qvals.clone().detach()
            chosen_action_qvals_clone.requires_grad = True
            target_max_qvals_clone = target_max_qvals.clone().detach()

            chosen_action_q_tot_vals = self.mixer(chosen_action_qvals_clone, batch["state"][:, :-1])
            target_max_q_tot_vals = self.target_mixer(target_max_qvals_clone, batch["state"][:, 1:])
            goal_target_max_qvals = self.target_mixer(target_ind_qvals, batch["state"][:, :-1])
        #####################################################################################################
        q_ind_tot_list = []
        for i in range(self.n_agents):
            target_qtot_per_agent = (goal_target_max_qvals / self.n_agents).squeeze()
            q_ind_tot_list.append(self.alpha * target_ind_qvals[:, :, i] + (1 - self.alpha) * target_qtot_per_agent)

        q_ind_tot = th.stack(q_ind_tot_list, dim=2)

        ddqn_qval_up_idx = th.max(q_ind_tot, dim=1)[1]  # Find out max Q value for t=1~T-1 (whole episode)

        explore_q_target = th.ones(target_ind_q.shape) / target_ind_q.shape[-1]
        explore_q_target = explore_q_target.to(device=self.device)

        ddqn_up_list = []
        distance_list = []
        for i in range(batch.batch_size):
            ddqn_up_list_subset = []
            distance_subset = []
            explore_loss_subset = []
            for j in range(self.n_agents):
                # For distance function

                cos = nn.CosineSimilarity(dim=-1, eps=1e-8)
                cos1 = nn.CosineSimilarity()
                goal_q = target_ind_q[i, ddqn_qval_up_idx[i][j], j, :].repeat(target_ind_q.shape[1], 1)

                a = cos(target_ind_q[i, :, j, :], goal_q)
                b = cos1(target_ind_q[i, :, j, :], goal_q)

                similarity = 1 - cos(target_ind_q[i, :, j, :], goal_q)
                dist_obs = self.distance(observation[i, :, j, :])
                dist_og = self.distance(observation[i, ddqn_qval_up_idx[i][j], j, :])

                dist_loss = th.norm(dist_obs - dist_og.repeat(dist_obs.shape[0], 1), dim=-1) - similarity
                distance_loss = th.mean(dist_loss ** 2)
                distance_subset.append(distance_loss)
                ddqn_up_list_subset.append(observation[i, ddqn_qval_up_idx[i][j], j, :])

            distance1 = th.stack(distance_subset)
            distance_list.append(distance1)

            ddqn_up1 = th.stack(ddqn_up_list_subset)
            ddqn_up_list.append(ddqn_up1)

        distance_losses = th.stack(distance_list)

        mix_explore_distance_losses = self.dis * distance_losses

        ddqn_up = th.stack(ddqn_up_list)
        ddqn_up = ddqn_up.unsqueeze(dim=1)
        ddqn_up = ddqn_up.repeat(1, observation.shape[1], 1, 1)

        reward_ddqn_up = self.distance(observation) - self.distance(ddqn_up)

        intrinsic_reward_list = []
        for i in range(self.n_agents):
            intrinsic_reward_list.append(
                -th.norm(reward_ddqn_up[:, :, i, :], dim=2).reshape(batch.batch_size, observation.shape[1]))
        intrinsic_rewards_ind = th.stack(intrinsic_reward_list, dim=-1)  # (B,T,n_agents)

        intrinsic_rewards = th.zeros(rewards.shape).to(device=self.device)

        for i in range(self.n_agents):
            intrinsic_rewards += -th.norm(reward_ddqn_up[:, :, i, :], dim=2).reshape(batch.batch_size,
                                                                                     observation.shape[1],
                                                                                     1) / self.n_agents
        rewards += self.lam * intrinsic_rewards

        ####################################################################
        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_q_tot_vals  # (B,T,1)

        # Td-error
        td_error = (chosen_action_q_tot_vals - targets.detach())  # (B,T,1)
        td_error_1 = (chosen_action_q_tot_vals - reward - self.args.gamma * (1 - terminated) * target_max_q_tot_vals)

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        

        # Normal L2 loss, take mean over actual data
        mix_explore_distance_loss = mix_explore_distance_losses.mean()
        mixer_loss = (masked_td_error ** 2).sum() / mask.sum() + 0.001 * self.mix * mix_explore_distance_loss

        # Optimise
        self.mixer_optimiser.zero_grad()

        chosen_action_qvals_clone.retain_grad()  # the grad of qi

        chosen_action_q_tot_vals.retain_grad()  # the grad of qtot
        mixer_loss.backward()

        grad_l_qtot = chosen_action_q_tot_vals.grad.repeat(1, 1, self.args.n_agents) + 1e-8
        grad_l_qi = chosen_action_qvals_clone.grad
        grad_qtot_qi = th.clamp(grad_l_qi / grad_l_qtot, min=-10, max=10)  # (B,T,n_agents)

        mixer_grad_norm = th.nn.utils.clip_grad_norm_(self.mixer_params, self.args.grad_norm_clip)
        self.mixer_optimiser.step()

        q_rewards = self.cal_indi_reward(grad_qtot_qi, td_error, chosen_action_qvals, target_max_qvals,
                                         indi_terminated)  # (B,T,n_agents)(td_error_1)
        q_rewards_clone = q_rewards.clone().detach()
        

        q_targets = q_rewards_clone + self.args.gamma * (1 - indi_terminated) * target_max_qvals  # (B,T,n_agents)

        # Td-error
        q_td_error = (chosen_action_qvals - q_targets.detach())  # (B,T,n_agents)
        

        q_mask = batch["filled"][:, :-1].float().repeat(1, 1, self.args.n_agents)  # (B,T,n_agents)
        q_mask[:, 1:] = q_mask[:, 1:] * (1 - indi_terminated[:, :-1]) * (1 - terminated[:, :-1]).repeat(1, 1,
                                                                                                        self.args.n_agents)
     
        q_mask = q_mask.expand_as(q_td_error)
        q2_mask = th.cat((q_mask,q_mask),dim=0)

        masked_q_td_error = q_td_error * q_mask
        down_value = th.zeros((masked_q_td_error.shape[0],self.n_agents))
        up_value = th.zeros((masked_q_td_error.shape[0],self.n_agents))
        mean,std,up_1z,down_1z = self.sum_and_sig(masked_q_td_error,down_value,up_value)
        share_list = th.zeros((masked_q_td_error.shape[0],masked_q_td_error.shape[1],self.n_agents))
        for i in range(masked_q_td_error.shape[0]):
            for j in range(masked_q_td_error.shape[1]):
                for agent in range(self.n_agents):
                    if masked_q_td_error[i][j][agent] >= up_1z[i][agent] or masked_q_td_error[i][j][agent] <= down_1z[i][agent]:
                        share_list[i][j][agent] == masked_q_td_error[i][j][agent]
        jieshou_list = th.zeros(share_list.shape[0],masked_q_td_error.shape[1],self.n_agents)
        for i in range(masked_q_td_error.shape[0]):
            for j in range(masked_q_td_error.shape[1]):
                dim = 0
                for agent in range(self.n_agents):
                    if agent == dim:
                        jieshou_list[i][j][agent] == 0
                    else:
                        if share_list[i][j][dim] >= up_1z[i][agent] or share_list[i][j][dim] <= down_1z[i][agent]:
                            jieshou_list[i][j][agent] = share_list[i][j][dim]
                dim = dim + 1
        jieshou_list = jieshou_list.to(self.device)
        masked_q_td_error = th.cat((masked_q_td_error, jieshou_list), dim=0)

        q_selected_weight, selected_ratio = self.select_trajectory(masked_q_td_error.abs(), q2_mask, t_env)
        q_selected_weight = q_selected_weight.clone().detach()
        # 0-out the targets that came from padded data

        # Normal L2 loss, take mean over actual data
        q_loss = (masked_q_td_error ** 2 * q_selected_weight).sum() / q2_mask.sum()
        

        # Optimise
        self.q_optimiser.zero_grad()
        q_loss.backward()
        q_grad_norm = th.nn.utils.clip_grad_norm_(self.q_params, self.args.grad_norm_clip)
        self.q_optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("selected_ratio", selected_ratio, t_env)
            self.logger.log_stat("mixer_loss", mixer_loss.item(), t_env)
            self.logger.log_stat("mixer_grad_norm", mixer_grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("mixer_td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("mixer_target_mean", (targets * mask).sum().item() / mask_elems, t_env)

            # self.logger.log_stat("q_loss", q_loss.item(), t_env)
            # self.logger.log_stat("q_grad_norm", q_grad_norm, t_env)
            # q_mask_elems = q2_mask.sum().item()
            # self.logger.log_stat("q_td_error_abs", (masked_q_td_error.abs().sum().item() / q_mask_elems), t_env)
            # self.logger.log_stat("q_q_taken_mean", (chosen_action_qvals * q_mask).sum().item() / (q_mask_elems), t_env)
            # self.logger.log_stat("mixer_target_mean", (q_targets * q_mask).sum().item() / (q_mask_elems), t_env)
            # self.logger.log_stat("reward_i_mean", (q_rewards * q_mask).sum().item() / (q_mask_elems), t_env)
            # self.logger.log_stat("q_selected_weight_mean", (q_selected_weight * q2_mask).sum().item() / (q_mask_elems),
            #                      t_env)

            self.log_stats_t = t_env

    def sum_and_sig(self, masked_q_td_error, down_value, up_value):

        each_batch_mean = th.mean(masked_q_td_error,dim=1)  # 平均值 # torch.Size([32,3])
        each_batch_std = th.std(masked_q_td_error,dim=1)           # 每个batch时间步长动作的标准差 # torch.Size([32,3])

        for i in range(each_batch_mean.shape[0]):
            for j in range(self.n_agents):
                down_value[i][j] = each_batch_mean[i][j] - each_batch_std[i][j]
                up_value[i][j] = each_batch_mean[i][j] + each_batch_std[i][j]
        return each_batch_mean, each_batch_std,down_value,up_value  # torch.Size([32,3])

    # def cal_indi_reward(grad_qtot_qi, td_error, chosen_action_qvals, target_max_qvals):
    def cal_indi_reward(self, grad_qtot_qi, mixer_td_error, qi, target_qi, indi_terminated):
        # input: grad_qtot_qi (B,T,n_agents)  mixer_td_error (B,T,1)  qi (B,T,n_agents)  indi_terminated (B,T,n_agents)
        grad_td = th.mul(grad_qtot_qi, mixer_td_error.repeat(1, 1, self.args.n_agents))  # (B,T,n_agents)
        reward_i = - grad_td + qi - self.args.gamma * (1 - indi_terminated) * target_qi
        return reward_i

    def select_trajectory(self, td_error, mask, t_env):
        # td_error (B, T, n_agents)
        if self.args.warm_up:
            if t_env / self.args.t_max <= self.args.warm_up_ratio:
                selected_ratio = t_env * (self.args.selected_ratio_end - self.args.selected_ratio_start) / (
                        self.args.t_max * self.args.warm_up_ratio) + self.args.selected_ratio_start
            else:
                selected_ratio = self.args.selected_ratio_end
        else:
            selected_ratio = self.args.selected_ratio

        if self.args.selected == 'all':
            return th.ones_like(td_error).cuda(), selected_ratio
        elif self.args.selected == 'greedy':
            valid_num = mask.sum().item()
            selected_num = int(valid_num * selected_ratio)
            td_reshape = td_error.reshape(-1)
            sorted_td, _ = th.topk(td_reshape, selected_num)
            pivot = sorted_td[-1]
            weight = th.where(td_error >= pivot, th.ones_like(td_error), th.zeros_like(td_error))
            return weight, selected_ratio
        elif self.args.selected == 'greedy_weight':
            valid_num = mask.sum().item()
            selected_num = int(valid_num * selected_ratio)
            td_reshape = td_error.reshape(-1)
            sorted_td, _ = th.topk(td_reshape, selected_num)
            pivot = sorted_td[-1]
            weight = th.where(td_error >= pivot, td_error - pivot, th.zeros_like(td_error))
            norm_weight = weight / weight.max()
            return norm_weight, selected_ratio
        
        elif self.args.selected == 'PER_hard':
            memory_size = int(mask.sum().item())
            selected_num = int(memory_size * selected_ratio)
            return PER_Memory(self.args, td_error, mask).sample(selected_num), selected_ratio
        elif self.args.selected == 'PER_weight':
            memory_size = int(mask.sum().item())
            selected_num = int(memory_size * selected_ratio)
            return PER_Memory(self.args, td_error, mask).sample_weight(selected_num, t_env), selected_ratio

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.q_optimiser.state_dict(), "{}/q_opt.th".format(path))
        th.save(self.mixer_optimiser.state_dict(), "{}/mixer_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.q_optimiser.load_state_dict(th.load("{}/q_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.mixer_optimiser.load_state_dict(
            th.load("{}/mixer_opt.th".format(path), map_location=lambda storage, loc: storage))
