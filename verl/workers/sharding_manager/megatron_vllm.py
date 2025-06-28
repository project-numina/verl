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
This file contains a Megatron style Hybrid Engine that shares the weights of the actor with the inference engine.
"""

import inspect
import logging
import os

import torch
import torch.distributed
from megatron.core import parallel_state as mpu
from torch import nn
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from verl import DataProto
from verl.models.mcore.weight_converter import McoreToHFWeightConverterBase
from verl.protocol import all_gather_data_proto
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.debug.performance import simple_timer
from verl.utils.device import get_torch_device
from verl.utils.megatron_utils import (
    load_megatron_model_to_gpu,
    offload_megatron_model_to_cpu,
    per_tensor_generator,
)

from verl.utils.model import normalize_model_name
from verl.utils.vllm_utils import patch_vllm_moe_model_weight_loader

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


"""
Megatron Hybrid Engine:
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank 
   to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""


class MegatronVLLMShardingManager(BaseShardingManager):
    @check_device_is_available()
    def __init__(
        self,
        actor_module: nn.ModuleList,
        inference_engine: LLM,
        model_config,
        transformer_config,
        layer_name_mapping,
        weight_converter: McoreToHFWeightConverterBase,
        device_mesh,
        offload_param: bool = True,
    ):
        self.actor_module = actor_module
        self.inference_engine = inference_engine
        self.offload_param = offload_param

        # For AsyncLLM, inference_engine and model_runner are defer initialized in vLLMAsyncRollout.load_model
        if "vllm_v_0_6_3" in str(type(self.inference_engine)) or "vllm_v_0_5_4" in str(type(self.inference_engine)):
            # vLLM <= v0.6.3
            self.model_runner = self.inference_engine.llm_engine.model_executor.worker.model_runner if self.inference_engine else None
        else:
            # vLLM > v0.6.3
            self.model_runner = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner if self.inference_engine else None

        self.model_config = model_config
        self.transformer_config = transformer_config
        self.layer_name_mapping = layer_name_mapping
        self.weight_converter = weight_converter
        self.module = module
        # initialize groups for vllm inference
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.infer_tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        self.infer_tp_rank = vllm_ps.get_tensor_model_parallel_rank()
        self.infer_tp_group = vllm_ps.get_tensor_model_parallel_group()
        if vllm_version not in ("0.5.4", "0.6.3"):
            self.infer_tp_group = self.infer_tp_group.device_group
        self.train_tp_size = mpu.get_tensor_model_parallel_world_size()
        self.train_tp_rank = mpu.get_tensor_model_parallel_rank()
        self.train_tp_group = mpu.get_tensor_model_parallel_group()
        self.need_tp_reshard = self.train_tp_size != self.infer_tp_size
        self.train_tp_larger = self.train_tp_size > self.infer_tp_size

    def per_tensor_generator(self, convert_qkv_gate_up_by_simple_split=True):
        """
        convert_qkv_gate_up_by_simple_split is a parameter affected by the vLLM version.
        """
        from megatron.core import parallel_state as mpu

        pp_rank = mpu.get_pipeline_model_parallel_rank()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        vpp_size = len(self.actor_module)

        all_gather_group = self.train_tp_group
        all_gather_group_size = torch.distributed.get_world_size(group=all_gather_group)

        def tensor_generator():
            for scan_vpp_idx in range(vpp_size):
                yield from self.actor_module[scan_vpp_idx].named_parameters()

        # we need first make all rank get full model information
        meta_info = []
        for scan_vpp_idx in range(vpp_size):
            for idx, (name, _) in enumerate(self.actor_module[scan_vpp_idx].named_parameters()):
                meta_info.append((pp_rank, scan_vpp_idx, idx, name))

        obj_spec_output = [None] * mpu.get_pipeline_model_parallel_world_size()
        torch.distributed.all_gather_object(object_list=obj_spec_output, obj=meta_info, group=mpu.get_pipeline_model_parallel_group())
        layer_list_meta = [item for sublist in obj_spec_output for item in sublist]

        gen_func = tensor_generator()

        # lazy load tensor for full model
        for cur_pp_rank, scan_vpp_idx, idx, name in layer_list_meta:
            if self.model_config.tie_word_embeddings and ("output_layers" in name):
                import warnings

                warnings.warn("Current model sharing word and embedding weights, skip output layer conversion", stacklevel=2)
                continue
            if cur_pp_rank == pp_rank:
                try:
                    cur_name, cur_tensor = next(gen_func)
                except StopIteration:
                    cur_name, cur_tensor = None, None
                cur_name = normalize_model_name(name, cur_pp_rank, scan_vpp_idx, pp_size, vpp_size, self.model_config.num_hidden_layers)
            else:
                cur_tensor, cur_name = None, None

            # pp broadcast model tensor and name
            cur_name = broadcast_str_from_megatron_pp(cur_name)
            broad_pp_tensor = broadcast_from_megatron_pp(cur_tensor)

            # (xya): this is a hack to fix the name of the parameters
            while cur_name.startswith("module."):
                cur_name = cur_name[len("module.") :]

            # tp all gather
            if tp_utils.is_tensor_parallel_param(broad_pp_tensor):
                # allocate a new tensor with proper size
                if all_gather_group_size <= 1:
                    infer_params = [broad_pp_tensor]
                else:
                    infer_params = [torch.empty_like(broad_pp_tensor) for _ in range(all_gather_group_size)]
                    torch.distributed.all_gather(infer_params, broad_pp_tensor, group=mpu.get_tensor_model_parallel_group())
                infer_params = self.default_tp_concat_fn(cur_name, broad_pp_tensor, infer_params, self.model_config, convert_qkv_gate_up_by_simple_split)
            else:
                infer_params = broad_pp_tensor

            if vllm_version in ("0.4.2", "0.5.4", "0.6.3"):
                converted_names, converted_params = convert_megatron_model_to_transformers_model(
                    cur_name,
                    infer_params,
                    self.model_config,
                    self.train_tp_size,
                    0,  # no impact
                    convert_qkv_gate_up_by_trunk_concat=False,
                )  # defualt false
            else:
                if not isinstance(infer_params, list):
                    infer_params = [infer_params]
                converted_names, converted_params = self.weight_converter.convert_param(cur_name, infer_params)

            yield from zip(converted_names, converted_params)

    def default_tp_concat_fn(self, name, param, infer_params, model_config, convert_qkv_gate_up_by_simple_split=False):
        """
        name: name of the parameter
        param: training parameters
        infer_params (Iterable[torch.Tensor]): a iterator towards list of parameters all-gathered
          from train tp group (vllm 0.8.2) or micro-dp group (vllm <= 0.6.3)
        model_config: huggingface model_config
        TODO(zhangchi.usc1992): currently, the implementation is adhoc. We can move this function to the model
        definition so that it is model-agnostic. If the model doesn't implement this function,
        we can throw an error to force user disable TP HybridEngine.
        """
        if self.layer_name_mapping.get("qkv_layer_name") in name and "layer_norm" not in name:
            # if the tensor is qkv, for each param on tp, split into q, k, v
            # concat q, k, v separately.
            q_lst = []
            k_lst = []
            v_lst = []
            assert model_config.num_attention_heads % model_config.num_key_value_heads == 0
            num_q_per_kv = model_config.num_attention_heads // model_config.num_key_value_heads
            assert infer_params[0].shape[0] % (num_q_per_kv + 2) == 0, f"param '{name}' shape '{infer_params[0].shape}' dim0 is not divisible by {num_q_per_kv + 2}"
            kv_size_per_tp = infer_params[0].shape[0] // (num_q_per_kv + 2)
            split_size = [kv_size_per_tp * num_q_per_kv, kv_size_per_tp, kv_size_per_tp]
            for infer_param in infer_params:
                num_query_groups_per_partition = model_config.num_key_value_heads // self.train_tp_size
                for chunk in infer_param.chunk(num_query_groups_per_partition):
                    split_size = [
                        kv_size_per_tp * num_q_per_kv // num_query_groups_per_partition,
                        kv_size_per_tp // num_query_groups_per_partition,
                        kv_size_per_tp // num_query_groups_per_partition,
                    ]
                    q, k, v = chunk.split(split_size)
                    q_lst.append(q)
                    k_lst.append(k)
                    v_lst.append(v)
            q = torch.cat(q_lst, dim=0)
            k = torch.cat(k_lst, dim=0)
            v = torch.cat(v_lst, dim=0)
            infer_params = torch.cat((q, k, v), dim=0) if not convert_qkv_gate_up_by_simple_split else [q, k, v]

        elif self.layer_name_mapping.get("gate_proj_layer_name") in name:
            # if the tensor is gate and proj
            gate_lst = []
            up_lst = []
            for infer_param in infer_params:
                gate, up = infer_param.chunk(2)
                gate_lst.append(gate)
                up_lst.append(up)
            gate = torch.cat(gate_lst, dim=0)
            up = torch.cat(up_lst, dim=0)
            infer_params = torch.cat((gate, up), dim=0) if not convert_qkv_gate_up_by_simple_split else [gate, up]

        elif "mlp.experts.linear_fc2.weight" in name:  # moe
            infer_params = torch.cat(infer_params, dim=1)

        self.torch_random_states = get_torch_device().get_rng_state()
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)
        else:
            # concat tensor
            infer_params = torch.cat(infer_params, dim=tp_utils.get_tensor_parallel_partition_dim(param))

        return infer_params

    def _post_process_params(self, params, convert_qkv_gate_up_by_simple_split=False):
        """
        For each param, if it is a tp-splited param, we all-gather from train tp group
        """
        # here the params are in train tp format. we iterate params and all-gather
        # TODO(zhangchi.usc1992) We can consider copy non-tp weight to another infer buffer.
        # In this way, all the params in the original memory_buffers and can be offload.
        all_gather_group = self.train_tp_group
        all_gather_group_size = torch.distributed.get_world_size(group=all_gather_group)

        for name, param in params:
            if tp_utils.is_tensor_parallel_param(param):
                # allocate a new tensor with proper size
                if all_gather_group_size <= 1:
                    infer_params = [param]
                else:
                    infer_params = [torch.empty_like(param) for _ in range(all_gather_group_size)]
                    torch.distributed.all_gather(infer_params, param, group=all_gather_group)
                infer_params = self.default_tp_concat_fn(name, param, infer_params, self.model_config, convert_qkv_gate_up_by_simple_split)
            else:
                infer_params = param
            if vllm_version in ("0.4.2", "0.5.4", "0.6.3"):
                converted_names, converted_params = convert_megatron_model_to_transformers_model(
                    name,
                    infer_params,
                    self.model_config,
                    self.train_tp_size,
                    self.module.pp_models[0][0].config.num_query_groups,
                    convert_qkv_gate_up_by_trunk_concat=False,
                )
            else:
                if not isinstance(infer_params, list):
                    infer_params = [infer_params]
                converted_names, converted_params = self.weight_converter.convert_param(name, infer_params)
            yield from zip(converted_names, converted_params)

    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=logger)
    def __enter__(self):
        self.timing = {}
        with simple_timer("reshard", self.timing):
            get_torch_device().empty_cache()

            log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
            if self.offload_param:
                load_megatron_model_to_gpu(self.actor_module)

            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                per_tensor_param = per_tensor_generator(self.actor_module, self.model_config, self.weight_converter, self.transformer_config, self.layer_name_mapping, convert_qkv_gate_up_by_simple_split=False)
                self.inference_engine.sync_model_weights(per_tensor_param, load_format="megatron")
            else:
                # > 0.7.2
                if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                    self.inference_engine.wake_up(tags=["weights"])
                else:
                    self.inference_engine.wake_up()
                per_tensor_param = per_tensor_generator(
                    self.actor_module,
                    self.model_config,
                    self.weight_converter,
                    self.transformer_config,
                    self.layer_name_mapping,
                )
                model = self.model_runner.model
                patch_vllm_moe_model_weight_loader(model)
                loaded_params = model.load_weights(per_tensor_param)
                info = f"vLLM load weights, loaded_params: {len(loaded_params)}"
                logger.info(info)

            if self.offload_param:
                offload_megatron_model_to_cpu(self.actor_module)
            get_torch_device().empty_cache()

            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["kv_cache"])

            # important: need to manually set the random states of each tp to be identical.
            if self.device_mesh is not None:
                self.torch_random_states = get_torch_device().get_rng_state()
                get_torch_device().set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            self.inference_engine.offload_model_weights()
        else:
            self.inference_engine.sleep(level=1)
        for model in self.actor_module:
            model.train()

        get_torch_device().empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)

    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=logger)
    def preprocess_data(self, data: DataProto) -> DataProto:
        # DP_COMPUTE_PROTO: all training ranks are dp, the same as fsdp
        if self.infer_tp_size == 1:
            return data
        all_gather_data_proto(data, self.infer_tp_group)
        return data

    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        # DP_COMPUTE_PROTO: all training ranks are dp, the same as fsdp
        if self.infer_tp_size == 1:
            return data
        return data.chunk(chunks=self.infer_tp_size)[self.infer_tp_rank]
