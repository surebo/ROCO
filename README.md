```diff
- Please pay attention to the version of SC2 you are using for your experiments. 
- Performance is *not* always comparable between versions. 
- The results in SMAC (https://arxiv.org/abs/1902.04043) use SC2.4.6.2.69232 not SC2.4.10. 
```

# ROCO: Role-Oriented Communication for Efficient Multi-Agent Reinforcement Learning

## Installation instructions

Install Python packages

```shell
# require Anaconda 3 or Miniconda 3
conda create -n pymarl python=3.8 -y
conda activate pymarl

bash install_dependecies.sh
```

Set up StarCraft II (2.4.10) and SMAC:

```shell
bash install_sc2.sh
```

This will download SC2.4.10 into the 3rdparty folder and copy the maps necessary to run over.

Set up Google Football:

```shell
bash install_gfootball.sh
```

## Command Line Tool

**Run an experiment**

```shell
# For SMAC
python3 src/main.py --config=roco --env-config=sc2 with n_role_clusters=5 t_max=3005000 env_args.map_name=MMM
```

```shell
# For Google Football
# map_name: academy_counterattack_easy, academy_3_vs_1_with_keeper
python3 src/main.py --config=roco --env-config=gfootball with env_args.map_name=academy_counterattack_hard env_args.num_agents=4
```

The config files act as defaults for an algorithm or environment.

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

**Kill all training processes**

```shell
# all python and game processes of current user will quit.
bash clean.sh
```

