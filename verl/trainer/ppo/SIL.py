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

    def replace(self, rollout_batch: DataProto):
        """
        Replace failed rollouts in the batch with successful ones from the database.
        Args:
            rollout_batch (DataProto): Batch of rollouts to replace.

        Returns:
            ids_to_recompute (list): List of indices in the batch that have been modified.
            ids_to_keep (list): List of indices in the batch that can be kept as is.
        """

        ids_to_recompute = []
        ids_to_keep = []

        for idx in range(len(rollout_batch)):
            rollout_item = rollout_batch[idx]
            prompt_idx = rollout_item.non_tensor_batch["index"]
            # Check if the reward is below the threshold
            if rollout_item.batch["acc"] < self.reward_threshold:
                # Replace with a random successful rollout from the database
                if self._buckets[prompt_idx]:
                    replacement = random.choice(list(self._buckets[prompt_idx]))
                    for key in rollout_item.batch.keys():
                        rollout_item.batch[key] = replacement.batch[key]
                    for key in rollout_item.non_tensor_batch.keys():
                        rollout_item.non_tensor_batch[key] = replacement.non_tensor_batch[key]

                    ids_to_recompute.append(idx)
                # No successful rollout in the database, keep the original
                else:
                    ids_to_keep.append(idx)

            # Reward is above the threshold, keep the original
            else:
                ids_to_keep.append(idx)

        return ids_to_recompute, ids_to_keep
