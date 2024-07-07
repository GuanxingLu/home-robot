# Home Assistant

# Install

Please follow the official home-robot repo to install the dependences: https://github.com/facebookresearch/home-robot

# Usage

```bash
conda activate home-robot
cd $HOME_ROBOT_ROOT
```

1. Collect the data
```bash
HABITAT_SIM_LOG=quiet CUDA_VISIBLE_DEVICES=6 python projects/habitat_ovmm/eval_baselines_agent.py --env_config projects/habitat_ovmm/configs/env/hssd_demo.yaml --data_dir data/datasets/rl_agent habitat.task.place_init=True habitat.dataset.split="train" habitat.environment.max_episode_steps=200
```

2. Train and merge the llava model

2. Test the model
```bash
HABITAT_SIM_LOG=quiet CUDA_VISIBLE_DEVICES=7 python projects/habitat_ovmm/eval_baselines_agent.py --env_config projects/habitat_ovmm/configs/env/hssd_demo.yaml --baseline_config_path projects/habitat_ovmm/configs/agent/llava_agent.yaml --data_dir data/datasets/llava_agent habitat.task.place_init=True habitat.dataset.split="train" habitat.environment.max_episode_steps=200
```
