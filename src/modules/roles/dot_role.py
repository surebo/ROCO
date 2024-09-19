import torch.nn as nn
import torch.nn.functional as F
import torch as th
import torch.distributions as D


class DotRole(nn.Module):
    def __init__(self, args):
        super(DotRole, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.latent_dim = args.action_latent_dim
        NN_HIDDEN_SIZE = args.nn_hidden_size
        activation_func = nn.LeakyReLU()
        
        self.q_fc = nn.Linear(args.rnn_hidden_dim, self.latent_dim)

        self.msg_net = nn.Sequential(
        nn.Linear(args.rnn_hidden_dim + self.latent_dim, NN_HIDDEN_SIZE),
        activation_func,
        nn.Linear(NN_HIDDEN_SIZE, self.n_actions)
        )

        self.w_key = nn.Linear(args.rnn_hidden_dim, args.attention_dim)
        self.w_query = nn.Linear(self.latent_dim, args.attention_dim)
        self.action_space = th.ones(self.n_actions).to(args.device)

    def forward(self, h, action_latent, test_mode=False):
        role_key = self.q_fc(h)  # [bs, latent_dim]
        role_key = role_key.unsqueeze(-1)  # [bs, latent_dim, 1]
        action_latent_reshaped = action_latent.unsqueeze(0).repeat(role_key.shape[0], 1, 1)
        q = th.bmm(action_latent_reshaped, role_key).squeeze(-1)  # [bs, n_actions]
        h_repeat = h.unsqueeze(1).repeat(1, self.n_actions, 1)  # [bs, n_actions, rnn_hidden_dim]
        msg = self.msg_net(th.cat([h_repeat, action_latent_reshaped], dim=-1))  # [bs, n_actions, n_actions]

        key = self.w_key(h).unsqueeze(1)  # [bs, 1, attention_dim]
        query = self.w_query(action_latent).unsqueeze(0)  # [1, n_actions, attention_dim]
        key = key.expand(-1, self.n_actions, -1)  # [bs, n_actions, attention_dim]
        query = query.expand(key.shape[0], -1, -1)  # [bs, n_actions, attention_dim]

        alpha = th.bmm(key / (self.args.attention_dim ** (1/2)), query.transpose(1, 2))  # [bs, n_actions, n_actions]
        alpha = F.softmax(alpha, dim=-1)  # [bs, n_actions, n_actions]

        if test_mode:
            alpha[alpha < (0.25 * 1 / self.n_actions)] = 0

        gated_msg = alpha * msg  # [bs, n_actions, n_actions]
        return_q = q + gated_msg.sum(dim=1)  # [bs, n_actions]

        return return_q

    def update_action_space(self, new_action_space):
     self.action_space = th.Tensor(new_action_space).to(self.args.device).float()
