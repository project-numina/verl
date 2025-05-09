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
Extension for random proof cutting.
"""

import logging
import random
from typing import Dict, Any, Callable, Optional, Union

logger = logging.getLogger(__name__)

def apply_random_cut(
    example: Dict[str, Any],
    tokenizer,
    cut_function: Callable,
    prompt_key: str = "prompt",
    cut_ratio_min: float = 0.4,
    cut_ratio_max: float = 0.6,
    **kwargs
) -> Dict[str, Any]:
    """
    Wrapper to apply a random cut to an example.
    
    Args:
        example: The dataset example to modify
        tokenizer: The tokenizer to use for token-based cutting
        cut_function: The domain-specific function that implements the cutting logic
        prompt_key: Key for the prompt in the example
        cut_ratio_min: Minimum ratio for the cut point
        cut_ratio_max: Maximum ratio for the cut point
        **kwargs: Additional arguments to pass to the cut function
        
    Returns:
        Modified example with a random cut applied
    """
    try:
        cut_ratio = random.uniform(cut_ratio_min, cut_ratio_max)
        
        return cut_function(
            example=example,
            tokenizer=tokenizer,
            prompt_key=prompt_key, 
            cut_ratio=cut_ratio,
            **kwargs
        )
    except Exception as e:
        logger.warning(f"Error applying random cut: {str(e)}")
        return example


class RandomCutMixin:
    """
    Mixin class to add random cutting capabilities to RLHFDataset.
    Can be used to extend any dataset class in veRL.
    """
    
    def _apply_random_cuts(self, 
                           cut_config: Dict[str, Any], 
                           cut_function: Callable,
                           filter_function: Optional[Callable] = None):
        """
        Apply random cuts to dataset examples.
        
        Args:
            cut_config: Configuration for the random cuts
            cut_function: The domain-specific function to apply the cut
            filter_function: Optional function to filter examples
        """
        if not cut_config.get("enabled", False):
            logger.info("Random cuts are disabled")
            return
            
        logger.info(f"Applying random cuts with config: {cut_config}")
        
        if filter_function and cut_config.get("apply_only_to_filtered", True):
            pre_filter_len = len(self.dataframe)
            self.dataframe = self.dataframe.filter(
                lambda example: filter_function(example, prompt_key=self.prompt_key),
                num_proc=self.num_workers,
                desc="Filtering examples for random cut"
            )
            post_filter_len = len(self.dataframe)
            logger.info(f"After filtering: {post_filter_len} examples (filtered out {pre_filter_len - post_filter_len})")
        
        apply_percentage = cut_config.get("apply_percentage", 100)
        
        if apply_percentage < 100:
            logger.info(f"Applying random cuts to {apply_percentage}% of eligible examples")
            
            def conditional_apply(example):
                if random.random() * 100 < apply_percentage:
                    return apply_random_cut(
                        example=example,
                        tokenizer=self.tokenizer,
                        cut_function=cut_function,
                        prompt_key=self.prompt_key,
                        cut_ratio_min=cut_config.get("cut_ratio_min", 0.4),
                        cut_ratio_max=cut_config.get("cut_ratio_max", 0.6),
                        **cut_config.get("cut_function_kwargs", {})
                    )
                return example
                
            self.dataframe = self.dataframe.map(
                conditional_apply,
                num_proc=self.num_workers,
                desc="Applying random cuts selectively"
            )
        else:
            self.dataframe = self.dataframe.map(
                lambda example: apply_random_cut(
                    example=example,
                    tokenizer=self.tokenizer,
                    cut_function=cut_function,
                    prompt_key=self.prompt_key,
                    cut_ratio_min=cut_config.get("cut_ratio_min", 0.4),
                    cut_ratio_max=cut_config.get("cut_ratio_max", 0.6),
                    **cut_config.get("cut_function_kwargs", {})
                ),
                num_proc=self.num_workers,
                desc="Applying random cuts"
            )
            
        logger.info(f"Successfully applied random cuts to dataset with {len(self.dataframe)} examples")