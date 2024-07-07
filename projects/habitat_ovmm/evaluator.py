# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import json
import os
import time
from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import pandas as pd
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.utils.info_dict import extract_scalars_from_info
from omegaconf import DictConfig
from termcolor import cprint
from tqdm import tqdm
from utils.env_utils import create_ovmm_env_fn
from utils.metrics_utils import get_stats_from_episode_metrics

if TYPE_CHECKING:
    from habitat.core.dataset import BaseEpisode
    from habitat.core.vector_env import VectorEnv

    from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent
    from home_robot.core.abstract_agent import Agent


class EvaluationType(Enum):
    """Whether we run local or remote evaluation."""

    LOCAL = "local"
    LOCAL_VECTORIZED = "local_vectorized"
    REMOTE = "remote"


class OVMMEvaluator(PPOTrainer):
    """Class for creating vectorized environments, evaluating OpenVocabManipAgent on an episode dataset and returning metrics."""

    def __init__(self, eval_config: DictConfig, data_dir=None) -> None:
        self.metrics_save_freq = eval_config.EVAL_VECTORIZED.metrics_save_freq
        self.results_dir = os.path.join(
            eval_config.DUMP_LOCATION, "results", eval_config.EXP_NAME
        )
        self.videos_dir = eval_config.habitat_baselines.video_dir
        self.data_dir = data_dir
        if self.data_dir:
            os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)

        super().__init__(eval_config)

    def local_evaluate_vectorized(self, agent, num_episodes_per_env=10):
        self._init_envs(
            config=self.config, is_eval=True, make_env_fn=create_ovmm_env_fn
        )
        self._evaluate_vectorized(
            agent,
            self.envs,
            num_episodes_per_env=num_episodes_per_env,
        )

    def _summarize_metrics(self, episode_metrics: Dict) -> Dict:
        """Gets stats from episode metrics"""
        # convert to a dataframe
        episode_metrics_df = pd.DataFrame.from_dict(episode_metrics, orient="index")
        episode_metrics_df["start_idx"] = 0
        stats = get_stats_from_episode_metrics(episode_metrics_df)
        return stats

    def _print_summary(self, summary: dict):
        """Prints the summary of metrics"""
        print("=" * 50)
        print("Averaged metrics")
        print("=" * 50)
        for k, v in summary.items():
            print(f"{k}: {v}")
        print("=" * 50)

    def _check_set_planner_vis_dir(
        self, agent: "Agent", current_episode: "BaseEpisode"
    ):
        """
        Sets vis_dir for storing planner's debug visualisations if the agent has a planner.
        """
        if hasattr(agent, "planner"):
            agent.planner.set_vis_dir(
                current_episode.scene_id.split("/")[-1].split(".")[0],
                current_episode.episode_id,
            )

    def _evaluate_vectorized(
        self,
        agent: "OpenVocabManipAgent",
        envs: "VectorEnv",
        num_episodes_per_env=None,
    ):
        # The stopping condition is either specified through
        # num_episodes_per_env (stop after each environment
        # finishes a certain number of episodes)
        print(f"Running eval on {envs.number_of_episodes} episodes")

        if num_episodes_per_env is None:
            num_episodes_per_env = envs.number_of_episodes
        else:
            num_episodes_per_env = [num_episodes_per_env] * envs.num_envs

        episode_metrics = {}

        def stop():
            return all(
                [
                    episode_idxs[i] >= num_episodes_per_env[i]
                    for i in range(envs.num_envs)
                ]
            )

        start_time = time.time()
        episode_idxs = [0] * envs.num_envs
        obs = envs.call(["reset"] * envs.num_envs)

        agent.reset_vectorized()
        self._check_set_planner_vis_dir(agent, self.envs.current_episodes()[0])
        while not stop():
            current_episodes_info = self.envs.current_episodes()
            # TODO: Currently agent can work with only 1 env, Parallelize act across envs
            actions, infos, _ = zip(*[agent.act(ob) for ob in obs])

            outputs = envs.call(
                ["apply_action"] * envs.num_envs,
                [{"action": a, "info": i} for a, i in zip(actions, infos)],
            )

            obs, dones, hab_infos = [list(x) for x in zip(*outputs)]
            for e, (done, info, hab_info) in enumerate(zip(dones, infos, hab_infos)):
                episode_key = (
                    f"{current_episodes_info[e].scene_id.split('/')[-1].split('.')[0]}_"
                    f"{current_episodes_info[e].episode_id}"
                )
                if episode_key not in episode_metrics:
                    episode_metrics[episode_key] = {}
                # Record metrics after each skill finishes. This is useful for debugging.
                if "skill_done" in info and info["skill_done"] != "":
                    metrics = extract_scalars_from_info(hab_info)
                    metrics_at_skill_end = {
                        f"{info['skill_done']}." + k: v for k, v in metrics.items()
                    }
                    episode_metrics[episode_key] = {
                        **metrics_at_skill_end,
                        **episode_metrics[episode_key],
                    }
                    if "goal_name" in episode_metrics[episode_key]:
                        episode_metrics[episode_key]["goal_name"] = info["goal_name"]
                if done:  # environment times out
                    metrics = extract_scalars_from_info(hab_info)
                    if episode_idxs[e] < num_episodes_per_env[e]:
                        metrics_at_episode_end = {
                            f"END." + k: v for k, v in metrics.items()
                        }
                        episode_metrics[episode_key] = {
                            **metrics_at_episode_end,
                            **episode_metrics[episode_key],
                        }
                        if "goal_name" in episode_metrics[episode_key]:
                            episode_metrics[episode_key]["goal_name"] = info[
                                "goal_name"
                            ]
                        episode_idxs[e] += 1
                        print(
                            f"Episode indexes {episode_idxs[e]} / {num_episodes_per_env[e]} "
                            f"after {round(time.time() - start_time, 2)} seconds"
                        )
                    if len(episode_metrics) % self.metrics_save_freq == 0:
                        aggregated_metrics = self._aggregate_metrics(episode_metrics)
                        self._write_results(episode_metrics, aggregated_metrics)
                    if not stop():
                        obs[e] = envs.call_at(e, "reset")
                        agent.reset_vectorized_for_env(e)
                        self._check_set_planner_vis_dir(
                            envs, envs.current_episodes()[e]
                        )

        envs.close()

        aggregated_metrics = self._aggregate_metrics(episode_metrics)
        self._write_results(episode_metrics, aggregated_metrics)

        average_metrics = self._summarize_metrics(episode_metrics)
        self._print_summary(average_metrics)

        return average_metrics

    def _aggregate_metrics(self, episode_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Aggregates metrics tracked by environment."""
        aggregated_metrics = defaultdict(list)
        metrics = set(
            [
                k
                for metrics_per_episode in episode_metrics.values()
                for k in metrics_per_episode
                if k != "goal_name"
            ]
        )
        for v in episode_metrics.values():
            for k in metrics:
                if k in v:
                    aggregated_metrics[f"{k}/total"].append(v[k])

        aggregated_metrics = dict(
            sorted(
                {
                    k2: v2
                    for k1, v1 in aggregated_metrics.items()
                    for k2, v2 in {
                        f"{k1}/mean": np.mean(v1),
                        f"{k1}/min": np.min(v1),
                        f"{k1}/max": np.max(v1),
                    }.items()
                }.items()
            )
        )

        return aggregated_metrics

    def _write_results(
        self, episode_metrics: Dict[str, Dict], aggregated_metrics: Dict[str, float]
    ) -> None:
        """Writes metrics tracked by environment to a file."""
        with open(f"{self.results_dir}/aggregated_results.json", "w") as f:
            json.dump(aggregated_metrics, f, indent=4)
        with open(f"{self.results_dir}/episode_results.json", "w") as f:
            json.dump(episode_metrics, f, indent=4)

    def local_evaluate(
        self, agent: "Agent", num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluates the agent in the local environment.

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """

        def preprocess_obs(observations):
            observation_save = observations.to_dict()
            observation_save["task_observations"]["object_embedding"] = None
            return observation_save

        def preprocess_info(info):
            info_save = copy.deepcopy(info)
            info_save["semantic_frame"] = None
            info_save["semantic_category_mapping"] = None
            return info_save

        def preprocess_action(action):
            """
            convert to str (.2f)
            """
            if isinstance(action, np.ndarray):
                action_save = action.tolist()
                # .2f
                action_save = [round(a, 4) for a in action_save]
            else:
                action_save = str(action)
            return action_save

        env_num_episodes = self._env.number_of_episodes
        if num_episodes is None:
            num_episodes = env_num_episodes
        else:
            assert num_episodes <= env_num_episodes, (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(num_episodes, env_num_episodes)
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        episode_metrics: Dict = {}

        count_episodes: int = 0

        pbar = tqdm(total=num_episodes)
        while count_episodes < num_episodes:
            observations, done = self._env.reset(), False
            current_episode = self._env.get_current_episode()
            agent.reset()
            self._check_set_planner_vis_dir(agent, current_episode)

            current_episode_key = (
                f"{current_episode.scene_id.split('/')[-1].split('.')[0]}_"
                f"{current_episode.episode_id}"
            )
            current_episode_metrics = {}
            expert_data = []
            skill_start_idx = 0
            step = 0
            fall_wait = False
            while not done:

                sample = {
                    "obs_data": None,
                    "action_data": None,
                    "info_data": None,
                    "step": 0,
                }

                action, info, _ = agent.act(
                    observations
                )  # e.g., action: DiscreteNavigationAction
                # print(f"action:{action}, step: {info['timestep']}")

                if self.data_dir:
                    if info["curr_skill"] != "FALL_WAIT":
                        sample["obs_data"] = preprocess_obs(observations)
                        sample["step"] = step
                    elif info["curr_skill"] == "FALL_WAIT" and not fall_wait:
                        sample["obs_data"] = preprocess_obs(observations)
                        sample["step"] = step

                observations, done, hab_info = self._env.apply_action(action, info)
                # print(hab_info)

                # TODO: if not fall wait action, save the data
                if self.data_dir:
                    if info["curr_skill"] != "FALL_WAIT":
                        # sample['action_data'] = preprocess_action(action)
                        sample["action_raw_data"] = preprocess_action(action)
                        sample["action_float_data"] = preprocess_action(
                            hab_info["action"]
                        )
                        sample["info_data"] = preprocess_info(info)
                        expert_data.append(sample)
                        # cprint(f"action_float_data: {sample['action_float_data']}", 'cyan')
                    elif info["curr_skill"] == "FALL_WAIT" and not fall_wait:
                        fall_wait = True
                        sample["action_raw_data"] = preprocess_action(action)
                        sample["action_float_data"] = preprocess_action(
                            hab_info["action"]
                        )
                        sample["info_data"] = preprocess_info(info)
                        expert_data.append(sample)
                        # cprint(f"fall wait, {action}, {hab_info['action']}", 'cyan')

                step += 1

                if "skill_done" in info and info["skill_done"] != "":
                    metrics = extract_scalars_from_info(hab_info)
                    metrics_at_skill_end = {
                        f"{info['skill_done']}." + k: v for k, v in metrics.items()
                    }
                    current_episode_metrics = {
                        **metrics_at_skill_end,
                        **current_episode_metrics,
                    }
                    if "goal_name" in info:
                        current_episode_metrics["goal_name"] = info["goal_name"]

            metrics = extract_scalars_from_info(hab_info)
            metrics_at_episode_end = {"END." + k: v for k, v in metrics.items()}
            current_episode_metrics = {
                **metrics_at_episode_end,
                **current_episode_metrics,
            }
            if "goal_name" in info:
                current_episode_metrics["goal_name"] = info["goal_name"]

            episode_metrics[current_episode_key] = current_episode_metrics
            if len(episode_metrics) % self.metrics_save_freq == 0:
                aggregated_metrics = self._aggregate_metrics(episode_metrics)
                self._write_results(episode_metrics, aggregated_metrics)
                average_metrics = self._summarize_metrics(episode_metrics)
                self._print_summary(average_metrics)

            # The task is considered successful if the agent places the object without robot collisions
            overall_success = (
                current_episode_metrics["END.robot_collisions.robot_scene_colls"] == 0
            ) * (current_episode_metrics["END.ovmm_place_success"] == 1)

            cprint(
                f"Ep {current_episode_key}, success: {overall_success}, col: {current_episode_metrics['END.robot_collisions.robot_scene_colls']}, does_want_terminate: {current_episode_metrics['END.does_want_terminate']}, inst: {None}",
                "green" if overall_success else "red",
            )

            # Save demo
            # if self.data_dir:
            if self.data_dir and overall_success:
                cprint(f"Saving data for episode {current_episode_key}", "green")
                import pickle

                data_episode_path = os.path.join(self.data_dir, current_episode_key)
                os.makedirs(data_episode_path, exist_ok=True)
                with open(os.path.join(data_episode_path, "obs_data.pkl"), "wb") as f:
                    pickle.dump(expert_data, f)

                # save metric
                with open(os.path.join(data_episode_path, "misc.pkl"), "wb") as f:
                    pickle.dump(current_episode_metrics, f)

            count_episodes += 1
            pbar.update(1)

        self._env.close()

        aggregated_metrics = self._aggregate_metrics(episode_metrics)
        self._write_results(episode_metrics, aggregated_metrics)

        average_metrics = self._summarize_metrics(episode_metrics)
        self._print_summary(average_metrics)

        return average_metrics

    def remote_evaluate(
        self, agent: "Agent", num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluates the agent in the remote environment.

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """

        # The modules imported below are specific to challenge remote evaluation.
        # These modules are not part of the home-robot repository.
        import pickle
        import time

        import grpc

        try:
            import evaluation_pb2
            import evaluation_pb2_grpc
        except ImportError:
            from home_robot_hw.utils.eval_ai import evaluation_pb2, evaluation_pb2_grpc

        # Wait for the remote environment to be up and running
        time.sleep(60)

        def grpc_dumps(entity):
            return pickle.dumps(entity)

        def grpc_loads(entity):
            return pickle.loads(entity)

        env_address_port = os.environ.get("EVALENV_ADDPORT", "localhost:8085")
        channel = grpc.insecure_channel(
            target=env_address_port,
            compression=grpc.Compression.Gzip,
            options=[
                (
                    "grpc.max_receive_message_length",
                    -1,
                )  # Unlimited message length that the channel can receive
            ],
        )
        stub = evaluation_pb2_grpc.EnvironmentStub(channel)

        stub.init_env(evaluation_pb2.Package())

        env_num_episodes = grpc_loads(
            stub.number_of_episodes(evaluation_pb2.Package()).SerializedEntity
        )
        if num_episodes is None:
            num_episodes = env_num_episodes
        else:
            assert num_episodes <= env_num_episodes, (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(num_episodes, env_num_episodes)
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        episode_metrics: Dict = {}

        count_episodes: int = 0

        pbar = tqdm(total=num_episodes)
        while count_episodes < num_episodes:
            observations, done = (
                grpc_loads(stub.reset(evaluation_pb2.Package()).SerializedEntity),
                False,
            )
            current_episode = grpc_loads(
                stub.get_current_episode(evaluation_pb2.Package()).SerializedEntity
            )
            agent.reset()
            self._check_set_planner_vis_dir(agent, current_episode)

            current_episode_key = (
                f"{current_episode.scene_id.split('/')[-1].split('.')[0]}_"
                f"{current_episode.episode_id}"
            )
            current_episode_metrics = {}

            while not done:
                action, info, _ = agent.act(observations)
                observations, done, hab_info = grpc_loads(
                    stub.apply_action(
                        evaluation_pb2.Package(
                            SerializedEntity=grpc_dumps((action, info))
                        )
                    ).SerializedEntity
                )

                # record metrics if the current skill finishes
                if hab_info is not None:
                    if "skill_done" in info and info["skill_done"] != "":
                        metrics = extract_scalars_from_info(hab_info)
                        metrics_at_skill_end = {
                            f"{info['skill_done']}." + k: v for k, v in metrics.items()
                        }
                        current_episode_metrics = {
                            **metrics_at_skill_end,
                            **current_episode_metrics,
                        }
                        if "goal_name" in info:
                            current_episode_metrics["goal_name"] = info["goal_name"]

            metrics = extract_scalars_from_info(hab_info)
            metrics_at_episode_end = {"END." + k: v for k, v in metrics.items()}
            current_episode_metrics = {
                **metrics_at_episode_end,
                **current_episode_metrics,
            }
            if "goal_name" in info:
                current_episode_metrics["goal_name"] = info["goal_name"]

            episode_metrics[current_episode_key] = current_episode_metrics
            if len(episode_metrics) % self.metrics_save_freq == 0:
                aggregated_metrics = self._aggregate_metrics(episode_metrics)
                self._write_results(episode_metrics, aggregated_metrics)

            count_episodes += 1
            pbar.update(1)

        stub.close(evaluation_pb2.Package())
        stub.evalai_update_submission(evaluation_pb2.Package())

        aggregated_metrics = self._aggregate_metrics(episode_metrics)
        self._write_results(episode_metrics, aggregated_metrics)

        average_metrics = self._summarize_metrics(episode_metrics)
        self._print_summary(average_metrics)

        return average_metrics

    def evaluate(
        self,
        agent: "Agent",
        num_episodes: Optional[int] = None,
        evaluation_type: str = "local",
    ) -> Dict[str, float]:
        r"""..

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the
            evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """
        if evaluation_type == EvaluationType.LOCAL.value:
            self._env = create_ovmm_env_fn(self.config)
            return self.local_evaluate(agent, num_episodes)
        elif evaluation_type == EvaluationType.LOCAL_VECTORIZED.value:
            self._env = create_ovmm_env_fn(self.config)
            return self.local_evaluate_vectorized(agent, num_episodes)
        elif evaluation_type == EvaluationType.REMOTE.value:
            self._env = None
            return self.remote_evaluate(agent, num_episodes)
        else:
            raise ValueError(
                "Invalid evaluation type. Please choose from 'local', 'local_vectorized', 'remote'"
            )
