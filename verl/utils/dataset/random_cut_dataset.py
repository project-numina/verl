import logging
import random
import traceback
import importlib
import sys
import os
from typing import Dict, Any

from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)

class RandomCutDataset(RLHFDataset):
    """
    Extended RLHFDataset that supports applying random cuts to content.
    """
    
    def _read_files_and_tokenize(self):
        super()._read_files_and_tokenize()
        
        random_cut_config = self.config.get("random_cut", {})
        if not random_cut_config.get("enabled", False):
            return
            
        cut_function_module = random_cut_config.get("cut_function_module")
        cut_function_name = random_cut_config.get("cut_function_name")
        filter_function_name = random_cut_config.get("filter_function_name")
        
        if not cut_function_module or not cut_function_name:
            logger.warning("Random cut is enabled but no cut function is specified")
            return
            
        try:
            logger.info(f"Importing module {cut_function_module}")
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            module = importlib.import_module(cut_function_module)
            logger.info(f"Successfully imported {cut_function_module}")
            
            cut_function = getattr(module, cut_function_name)
            filter_function = getattr(module, filter_function_name) if filter_function_name else None
            
            if filter_function and random_cut_config.get("apply_only_to_filtered", True):
                pre_filter_len = len(self.dataframe)
                logger.info(f"Filtering {pre_filter_len} examples")
                
                def robust_filter(example):
                    try:
                        return filter_function(example, prompt_key=self.prompt_key)
                    except Exception as e:
                        logger.debug(f"Filter error: {str(e)}")
                        return False
                
                self.dataframe = self.dataframe.filter(
                    robust_filter,
                    num_proc=self.num_workers,
                    desc="Filtering examples for random cut"
                )
                
                post_filter_len = len(self.dataframe)
                logger.info(f"After filtering: {post_filter_len} examples (filtered out {pre_filter_len - post_filter_len})")
            
            logger.info(f"Applying random cuts to {len(self.dataframe)} examples")
            
            cut_ratio_min = random_cut_config.get("cut_ratio_min", 0.4)
            cut_ratio_max = random_cut_config.get("cut_ratio_max", 0.6)
            cut_function_kwargs = random_cut_config.get("cut_function_kwargs", {})
            
            # Process all examples individually to avoid issues with batching
            processed_examples = []
            
            examples = list(self.dataframe)
            total = len(examples)
            
            for i, example in enumerate(examples):
                if i % 1000 == 0:
                    logger.info(f"Processing example {i}/{total}")
                
                try:
                    cut_ratio = random.uniform(cut_ratio_min, cut_ratio_max)
                    
                    processed = cut_function(
                        example=example.copy(),
                        tokenizer=self.tokenizer,
                        prompt_key=self.prompt_key,
                        cut_ratio=cut_ratio,
                        **cut_function_kwargs
                    )
                    
                    if "_random_cut_metadata" in processed:
                        metadata = processed.pop("_random_cut_metadata")
                        processed["cut_ratio"] = metadata.get("cut_ratio", cut_ratio)
                        processed["truncated_proof_length"] = len(metadata.get("truncated_proof", "").split())
                        processed["completion_proof_length"] = len(metadata.get("completion_proof", "").split())
                        processed["is_cut_proof"] = True
                    elif "random_cut_info" in processed:
                        metadata = processed.pop("random_cut_info")
                        processed["cut_ratio"] = metadata.get("cut_ratio", cut_ratio)
                        processed["truncated_proof_length"] = len(metadata.get("truncated_proof", "").split())
                        processed["completion_proof_length"] = len(metadata.get("completion_proof", "").split())
                        processed["is_cut_proof"] = True
                    
                    processed_examples.append(processed)
                except Exception as e:
                    logger.warning(f"Error processing example {i}: {str(e)}")
                    processed_examples.append(example.copy())
            
            from datasets import Dataset
            self.dataframe = Dataset.from_list(processed_examples)
            
            cut_count = 0
            for example in self.dataframe:
                if example.get("is_cut_proof", False):
                    cut_count += 1
                    
            logger.info(f"Successfully applied random cuts to {cut_count} out of {len(self.dataframe)} examples")
            
        except Exception as e:
            logger.error(f"Error in random cut process: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())