#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy
import random
from collections import OrderedDict
from typing import Any, Tuple, Union

import gym.spaces as spaces
import numba
import numpy as np
import torch
from habitat.core.agent import Agent
from habitat.core.spaces import EmptySpace
from habitat.gym.gym_wrapper import (
    continuous_vector_action_to_hab_dict,
    create_action_space,
)
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    get_active_obs_transforms,
)
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat_baselines.utils.common import batch_obs
from omegaconf import OmegaConf

import home_robot.utils.pose as pu
from home_robot.agent.ovmm_agent.complete_obs_space import get_complete_obs_space
from home_robot.core.interfaces import (
    ContinuousFullBodyAction,
    ContinuousNavigationAction,
    DiscreteNavigationAction,
    Observations,
)
from home_robot.utils.constants import (
    MAX_DEPTH_REPLACEMENT_VALUE,
    MIN_DEPTH_REPLACEMENT_VALUE,
)
import sys
import os
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from termcolor import cprint
from PIL import Image


random_generator = np.random.RandomState()


@numba.njit
def _seed_numba(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    _seed_numba(seed)
    torch.random.manual_seed(seed)


def sample_random_seed():
    set_random_seed(random_generator.randint(2**32))


class LLaVAgent(Agent):
    """
    Abstract class for evaluation of a LLaVA policy/skill. Loads the trained skill and takes actions
    """

    def __init__(
        self,
        config,
        skill_config,
        device_id: int = 0,
        obs_spaces=None,
        action_spaces=None,
    ) -> None:
        # Observation and action spaces for the full task
        self.device_id = device_id
        self.device = (
            torch.device(f"cuda:{self.device_id}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.config = config
        # Read in the RL config (in hydra format)
        self.rl_config = get_habitat_config(skill_config.rl_config)

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True  # type: ignore

        # load LLaVA
        disable_torch_init()
        # model_path = '/mnt/disk_1/yiqin/ckpt/llava-v1.5-7b'   # originla llava
        model_path = '/mnt/disk_1/guanxing/LLaVA/checkpoints/llava-v1.5-7b-manip-lora-merge'
        model_base = None
        model_path = os.path.expanduser(model_path)
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, model_base, model_name)

        self.temperature = 0.2
        self.conv_mode = "llava_v1"


    def reset(self) -> None:
        pass

    def reset_vectorized(self):
        """Initialize agent state."""
        self.reset()

    def reset_vectorized_for_env(self, e: int):
        """Initialize agent state."""
        self.reset()

    def does_want_terminate(self, observations, action) -> bool:
        return False

    def act(
        self, observations: Observations, info
    ) -> Tuple[
        Union[
            ContinuousFullBodyAction,
            ContinuousNavigationAction,
            DiscreteNavigationAction,
        ],
        bool,
    ]:
        sample_random_seed()
        obs = observations.rgb.copy()

        inp = 'Specify the action of manipulating the object.'

        if obs is not None:
            # first message
            if self.model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # prompt = '<image>\nSpecify the action of manipulating the object.'
        # cprint(prompt, 'cyan')

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)

        image = Image.fromarray(obs)
        image_size = image.size
        image_tensor = process_images([image], self.image_processor, self.model.config) # [1, 3, 336, 336]
        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        max_retry_num = 5
        retry_num = 0
        while retry_num < max_retry_num:
            try:
                retry_num += 1
                with torch.inference_mode():
                    output_ids = self.model.generate(
                        input_ids,
                        images=image_tensor,
                        image_sizes=[image_size],
                        do_sample=True if self.temperature > 0 else False,
                        temperature=self.temperature,
                        max_new_tokens=1024,
                        use_cache=True,
                    )

                outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

                cprint(outputs, 'cyan')

                # Map policy controlled arm_action to complete arm_action space
                step_action = outputs.strip().replace('The action is ', '')
                step_action = eval(step_action)

                break

            except KeyboardInterrupt:
                sys.exit()
        
            except Exception as e:
                cprint('output is not available', 'red')

        # joints = 
        # step_action = ContinuousFullBodyAction(joints, xyt=xyt)

        cprint(step_action, 'cyan')
        step_action = np.array(step_action)

        return (
            step_action,
            info,
            self.does_want_terminate(observations, step_action),
        )
