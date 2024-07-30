"""
Newly created for StatML project
"""


from typing import Tuple
import numpy as np
import torch
from gops.trainer.buffer.prioritized_replay_buffer import PrioritizedReplayBuffer
from gops.trainer.sampler.base import Experience

__all__ = ["PrioritizedStratifiedReplayBuffer"]


class PrioritizedStratifiedReplayBuffer(PrioritizedReplayBuffer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha = kwargs.get("prioritized_replay_buffer_alpha", 0.6)
        self.category_num = kwargs["category_num"]
        self.sum_tree = np.zeros((2 * self.max_size - 1, self.category_num))
        self.categories_priority = (1 / (self.sum_tree[0] + self.epsilon)) ** self.alpha
        self.alpha_increment = 0.1

    def store(
        self,
        exp: Experience,
    ) -> None:
        super(PrioritizedReplayBuffer, self).store(exp)
        tree_idx = self.prev_ptr + self.max_size - 1
        next_info = exp.next_info
        self.sum_tree[tree_idx, :] = np.eye(self.category_num)[next_info["category"], :]
        self.update_tree(tree_idx)


    def add_batch(self, samples: list):
        super().add_batch(samples)
        self.categories_priority = (1 / (self.sum_tree[0] + self.epsilon)) ** self.alpha

    def update_tree(self, tree_idx):
        parent = (tree_idx - 1) // 2
        while True:
            left = 2 * parent + 1
            right = left + 1
            self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
            if parent == 0:
                break
            parent = (parent - 1) // 2


    def get_leaf(self, value: float) -> Tuple[int, float]:
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self.sum_tree):
                idx = parent
                break
            else:
                left_tree_value = self.sum_tree_backup[left]
                if left_tree_value == 0:
                    left_tree_value = self.sum_tree_backup[left] = self.sum_tree[left] @ self.categories_priority
                if value <= left_tree_value:
                    parent = left
                else:
                    value -= left_tree_value
                    parent = right
        return idx, self.sum_tree[idx] @ self.categories_priority

    def sample_batch(self, batch_size: int) -> dict:
        self.sum_tree_backup = np.zeros((2 * self.max_size - 1,))  # to avoid recalculation
        self.sum_tree_backup[0] = self.sum_tree[0] @ self.categories_priority
        segment = self.sum_tree_backup[0] / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)
        min_prob = np.min(self.categories_priority) / self.sum_tree_backup[0]
        max_weight = (min_prob * self.size) ** (-self.beta)
        
        values = np.random.uniform(np.arange(batch_size) * segment, np.arange(batch_size) * segment + segment)
        tree_idxes, priorities = zip(*map(self.get_leaf, values))
        tree_idxes = np.array(tree_idxes, dtype=np.int32)
        priorities = np.array(priorities)
        probs = priorities / self.sum_tree_backup[0]
        weights = (probs * self.size) ** (-self.beta) / max_weight

        idxes = tree_idxes - self.max_size + 1
        batch = self.buf.sample(idxes, batch_size, self.seq_len, add_noise = self.add_noise)
        batch["idx"] = torch.as_tensor(tree_idxes, dtype=torch.int32)
        batch["weight"] = torch.as_tensor(weights, dtype=torch.float32)
        return batch

    def update_batch(self, idxes: int, priorities: float) -> None:
        if isinstance(idxes, torch.Tensor):
            idxes = idxes.cpu().detach().numpy()
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.cpu().detach().numpy()
        priorities = (priorities + self.epsilon) ** self.alpha
        mask = self.sum_tree[idxes] > 0
        self.sum_tree[idxes][mask] = priorities
        self.min_tree[idxes] = priorities
        self.max_priority = max(self.max_priority, priorities.max())
        list(map(self.update_tree, idxes))

    
    def increase_alpha(self):
        self.alpha = min(self.alpha + self.alpha_increment, 0.8)

    def save_data_dist(self, path, iteration, replay_samples):
        import os   
        from matplotlib import pyplot as plt

        for folder in ["buffer", "sample"]:
            if folder == "buffer":
                category = self.sum_tree[0]
                yscale = 'log'
            else:
                category = self.sum_tree[replay_samples["idx"].cpu()].sum(0)
                yscale = 'linear'

            plt.figure()
            bar = plt.bar(np.arange(self.category_num), category, tick_label=['Non-terminal', 'Out of lane', 'Collision', 'Braking ', 'Left turn', 'Right turn'], color='skyblue')
            plt.bar_label(bar, [f'$N_0={int(category[0])}$', f'$N_1={int(category[1])}$', f'$N_2={int(category[2])}$', f'$N_3={int(category[3])}$', f'$N_4={int(category[4])}$', f'$N_5={int(category[5])}$'], fmt='%d')
            plt.yscale(yscale)
            plt.title(r"Category distribution")
            os.makedirs(path + f"/{folder}", exist_ok=True)
            plt.savefig(path + f"/{folder}/category dist_{iteration}.png")
            plt.close()