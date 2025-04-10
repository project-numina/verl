# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

import os
import ray
import hydra




from autoformalizer.data_utils.process_reasoning_text import extract_proof_from_text
from autoformalizer.clients.lean4_client import Lean4Client, batch_verify_proof

import re
import os
import random

client = Lean4Client(
    url=os.environ.get("LEAN4_API_URL"),
    api_key=os.environ.get("LEAN4_API_KEY"),
)

TIMEOUT=60
LEAN4_PROOF_PARSING_ERRORS = [
    "Theorem statement couldn't be parsed from statement.",
    "No proof found in the output."
]

def random_reward(data_sources, solution_strs, ground_truths, extra_infos):

    rws = [random.random() for _ in solution_strs]
    rws = [1.0 if x < 0.6 else 0.0 for x in rws]
    return rws


def format_reward(solution_str):
    pattern = r"<think>(.*?)</think>\n```lean4\n(.*?)\n```"
    return 1.0 if re.search(pattern, solution_str, re.DOTALL) else 0.0

def proof_rewards(lean4_proofs, timeout = TIMEOUT):

    samples = [
        {
            "uuid": str(idx),
            "proof_id": str(idx),
            "proof": proof,
        }
        for idx, proof in enumerate(lean4_proofs)
    ]

    results = batch_verify_proof(client, samples, timeout=timeout)

    rewards = []
    uuid_to_result = {result["uuid"]: result for result in results}
    for idx, proof in enumerate(lean4_proofs):
        result = uuid_to_result[str(idx)]
        if result.get("is_valid_no_sorry", False):
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


def formal_reasoning_reward(data_sources, solution_strs, ground_truths, extra_infos, proof_weight = 0.9, format_weight = 0.1, return_dict=False):

    normalizer = proof_weight + format_weight
    proof_weight /= normalizer
    format_weight /= normalizer

    formal_statements = [extra_info["formal_statement"] for extra_info in extra_infos]
    lean4_proofs = [extract_proof_from_text(solution_str, formal_statement) for solution_str, formal_statement in zip(solution_strs, formal_statements)]
    proof_rws = proof_rewards(lean4_proofs)
    format_rws = [format_reward(solution_str) for solution_str in solution_strs]
    scores = [proof_weight * proof_rw + format_weight * format_rw for proof_rw, format_rw in zip(proof_rws, format_rws)]

    if not return_dict:
        return scores
    
    rws = []
    for lean4_proof, score, format_rw, proof_rw in zip(lean4_proofs, scores, format_rws, proof_rws):
        rws.append({
            "score": score,
            "pred": lean4_proof,
            "acc": proof_rw,
            #"format_reward": format_rw,
            #"proof_reward": proof_rw
        })
    
    return rws


def get_custom_reward_fn(config):
    import importlib.util, sys
    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    # TODO(linjunrong.ocss884): this ENV is left for resolving SGLang conflict with ray devices
    # isolation, will solve in the future
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'VLLM_LOGGING_LEVEL': 'WARN'
            }
        })

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:

    def run(self, config):
        from verl.utils.fs import copy_to_local
        # print initial config
        from pprint import pprint
        from omegaconf import OmegaConf
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_tokenizer, hf_processor
        trust_remote_code = config.data.get('trust_remote_code', False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        # define worker classes
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray import RayWorkerGroup
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == 'megatron':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = 'global_pool'
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy == 'fsdp':
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == 'megatron':
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        #use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "naive")
        if reward_manager_name == 'naive':
            from verl.workers.reward_manager import NaiveRewardManager
            reward_manager_cls = NaiveRewardManager
        elif reward_manager_name == 'prime':
            from verl.workers.reward_manager import PrimeRewardManager
            reward_manager_cls = PrimeRewardManager
        elif reward_manager_name == 'batch':
            from verl.workers.reward_manager import BatchRewardManager
            reward_manager_cls = BatchRewardManager
        elif reward_manager_name == 'dapo':
            from verl.workers.reward_manager import DAPORewardManager
            reward_manager_cls = DAPORewardManager
        else:

            raise NotImplementedError

        compute_score = get_custom_reward_fn(config)
        reward_kwargs = dict(config.reward_model.get("reward_kwargs", {}))
        reward_fn = reward_manager_cls(tokenizer=tokenizer,
                                       num_examine=0,
                                       compute_score=formal_reasoning_reward,
                                       reward_fn_key=config.data.reward_fn_key,
                                       **reward_kwargs)

        # Note that we always use function-based RM for validation
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer,
                                           num_examine=1,
                                           compute_score=formal_reasoning_reward,
                                           reward_fn_key=config.data.reward_fn_key)
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = RayPPOTrainer(config=config,
                                tokenizer=tokenizer,
                                processor=processor,
                                role_worker_mapping=role_worker_mapping,
                                resource_pool_manager=resource_pool_manager,
                                ray_worker_group_cls=ray_worker_group_cls,
                                reward_fn=reward_fn,
                                val_reward_fn=val_reward_fn)
        trainer.init_workers()
        trainer.fit()


if __name__ == '__main__':
    main()
