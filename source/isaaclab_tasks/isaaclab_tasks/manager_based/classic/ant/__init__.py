# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment (similar to OpenAI Gym Ant-v2).
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Ant-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_env_cfg:AntEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)


# Ant contact-friction randomization experiment.
import gymnasium as _ant_friction_gym

_ant_friction_gym.register(
    id="Isaac-Ant-Friction-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            f"{__name__}.ant_friction_env_cfg:AntFrictionEnvCfg",
        "rsl_rl_cfg_entry_point":
            f"{__name__}.agents.rsl_rl_ppo_cfg:AntPPORunnerCfg",
    },
)


# Ant small-rough-terrain experiments.
import gymnasium as _ant_rough_gym

for _task_id, _cfg_name in (
    ("Isaac-Ant-RoughTrain-v0", "AntRoughTrainEnvCfg"),
    ("Isaac-Ant-RoughPreview-v0", "AntRoughPreviewEnvCfg"),
):
    _ant_rough_gym.register(
        id=_task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point":
                f"{__name__}.ant_rough_env_cfg:{_cfg_name}",
            "rsl_rl_cfg_entry_point":
                f"{__name__}.agents.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        },
    )


# Analytic ramp evaluation with terrain-relative height.
import gymnasium as _ant_ramp_gym

_ant_ramp_gym.register(
    id="Isaac-Ant-RampEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            f"{__name__}.ant_ramp_env_cfg:AntRampEvalEnvCfg",
        "rsl_rl_cfg_entry_point":
            f"{__name__}.agents.rsl_rl_ppo_cfg:AntPPORunnerCfg",
    },
)
