# Copyright 2025 Individual Contributor: Thibaut Barroyer
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

import os
import pickle
import random
from collections import defaultdict, deque
from typing import Dict

from verl import DataProto
from verl.protocol import DataProtoItem


class RolloutDatabase:
    """
    Stores last k successfull rollouts for every prompts.
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
        self._buckets: Dict[str, deque[DataProtoItem]] = defaultdict(lambda: deque(maxlen=self.k))
        pass

    def add(self, rollout_batch: DataProto):
        """
        Add a batch of rollouts to the database.
        Args:
            batch (DataProto): Batch of rollouts to add.
        """
        for idx in range(len(rollout_batch)):
            rollout_item = rollout_batch[idx]
            prompt_idx = rollout_item.non_tensor_batch["index"]
            # Check if the reward is above the threshold
            if rollout_item.batch["acc"] >= self.reward_threshold:
                self._buckets[prompt_idx].append(rollout_item)

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

        # group all rollouts by prompt index
        prompt_to_indices = defaultdict(list)
        for idx in range(len(rollout_batch)):
            prompt_idx = rollout_batch[idx].non_tensor_batch["index"]
            prompt_to_indices[prompt_idx].append(idx)

        for prompt_idx, indices in prompt_to_indices.items():
            # Check if all responses for that prompt are below the threshold
            all_below_threshold = all(rollout_batch[i].batch["acc"] < self.reward_threshold for i in indices)

            if all_below_threshold and self._buckets[prompt_idx]:
                # Replace only one of them with a successful sample
                to_replace_idx = indices[0]
                replacement = random.choice(list(self._buckets[prompt_idx]))

                rollout_item = rollout_batch[to_replace_idx]
                for key in rollout_item.batch.keys():
                    rollout_item.batch[key] = replacement.batch[key]
                for key in rollout_item.non_tensor_batch.keys():
                    rollout_item.non_tensor_batch[key] = replacement.non_tensor_batch[key]

                ids_to_recompute.append(to_replace_idx)
                # Keep the rest
                ids_to_keep.extend(indices[1:])
            else:
                # Keep all original rollouts
                ids_to_keep.extend(indices)

        return ids_to_recompute, ids_to_keep

    def save(self, filepath: str):
        """
        Save the current rollout database buckets to disk

        Args:
            filepath (str): Path to the file where the database should be saved.
        """
        with open(filepath, "wb") as f:
            serializable_buckets = {k: list(v) for k, v in self._buckets.items()}
            pickle.dump(serializable_buckets, f)

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
            self._buckets = defaultdict(lambda: deque(maxlen=self.k))
            for k, v in data["buckets"].items():
                self._buckets[k] = deque(v, maxlen=self.k)
