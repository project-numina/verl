import os
import pickle
import random
from collections import defaultdict
from typing import Dict

from verl import DataProto


class RolloutDatabase:
    """
    Stores last k successful rollouts for every prompt.
    Can be used for replay or self-imitation learning.
    Can persist to disk and reload lazily using memory-mapped files if needed.
    """

    def __init__(self, k: int = 10, reward_threshold: float = 0.5):
        """
        Args:
            k (int): Number of rollouts to store for each prompt.
            reward_threshold (float): Minimum reward to store a rollout.
        """
        self.k = k
        self.reward_threshold = reward_threshold
        self._buckets: Dict[str, list] = defaultdict(list)

    def add(self, rollout_batch: DataProto):
        """
        Add a batch of rollouts to the database.
        Args:
            rollout_batch (DataProto): Batch of rollouts to add.
        """
        for idx in range(len(rollout_batch)):
            rollout_item = rollout_batch[idx]
            prompt_idx = rollout_item.non_tensor_batch["index"]
            if rollout_item.batch["acc"] >= self.reward_threshold:
                item = {
                    "responses": rollout_item.batch["responses"],
                    "response_mask": rollout_item.batch["response_mask"],
                    "token_level_scores": rollout_item.batch["token_level_scores"],
                    "acc": rollout_item.batch["acc"],
                    "nt_acc": rollout_item.non_tensor_batch["acc"],
                    "nt_score": rollout_item.non_tensor_batch["score"],
                }
                bucket = self._buckets[prompt_idx]
                bucket.append(item)
                if len(bucket) > self.k:
                    bucket.pop(0)  # Remove oldest

                print(f"Added rollout for prompt {prompt_idx}: {item['nt_acc']} in bucket of size {len(bucket)}")

    def replace_one_if_all_failed(self, rollout_batch: DataProto):
        """
        Replace a single failed rollout in the batch with a successful one from the database,
        but only if *all* rollouts for a prompt are below the reward threshold.

        Args:
            rollout_batch (DataProto): Batch of rollouts to replace. Modified in place.

        Returns:
            ids_to_recompute (list): List of indices in the batch that have been modified.
            ids_to_keep (list): List of indices in the batch that can be kept as is.
        """
        ids_to_recompute = []
        ids_to_keep = []
        prompt_to_indices = defaultdict(list)

        for idx in range(len(rollout_batch)):
            prompt_idx = rollout_batch[idx].non_tensor_batch["index"]
            prompt_to_indices[prompt_idx].append(idx)

        for prompt_idx, indices in prompt_to_indices.items():
            all_below_threshold = all(rollout_batch[i].batch["acc"] < self.reward_threshold for i in indices)

            if all_below_threshold and self._buckets[prompt_idx]:
                to_replace_idx = indices[0]
                replacement = random.choice(self._buckets[prompt_idx])

                rollout_batch.batch["responses"][to_replace_idx] = replacement["responses"].cpu().numpy()
                rollout_batch.batch["response_mask"][to_replace_idx] = replacement["response_mask"].cpu().numpy()
                rollout_batch.batch["acc"][to_replace_idx] = replacement["acc"].cpu().numpy()
                rollout_batch.batch["token_level_scores"][to_replace_idx] = replacement["token_level_scores"].cpu().numpy()

                rollout_batch.non_tensor_batch["score"][to_replace_idx] = replacement["nt_score"]
                rollout_batch.non_tensor_batch["acc"][to_replace_idx] = replacement["nt_acc"]

                ids_to_recompute.append(to_replace_idx)
                ids_to_keep.extend(indices[1:])
            else:
                ids_to_keep.extend(indices)

        return ids_to_recompute, ids_to_keep

    def save(self, filepath: str):
        """
        Save the current rollout database buckets to disk

        Args:
            filepath (str): Path to the file where the database should be saved.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self._buckets, f)

    def load(self, filepath: str):
        """
        Load a rollout database from disk.

        Args:
            filepath (str): Path to the saved database file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No saved database found at: {filepath}")

        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self._buckets = defaultdict(list, {k: v[: self.k] for k, v in data.items()})
