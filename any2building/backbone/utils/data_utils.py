from typing import *
import math
import torch
import numpy as np
from torch.utils.data import Sampler, Dataset, DataLoader, DistributedSampler
import torch.distributed as dist


def recursive_to_device(
    data: Any,
    device: torch.device,
    non_blocking: bool = False,
) -> Any:
    """
    Recursively move all tensors in a data structure to a device.
    """
    if hasattr(data, "to"):
        return data.to(device, non_blocking=non_blocking)
    elif isinstance(data, (list, tuple)):
        return type(data)(recursive_to_device(d, device, non_blocking) for d in data)
    elif isinstance(data, dict):
        return {k: recursive_to_device(v, device, non_blocking) for k, v in data.items()}
    else:
        return data


def load_balanced_group_indices(
    load: List[int],
    num_groups: int,
    equal_size: bool = False,
) -> List[List[int]]:
    """
    Split indices into groups with balanced load.
    """
    if equal_size:
        group_size = len(load) // num_groups
    indices = np.argsort(load)[::-1]
    groups = [[] for _ in range(num_groups)]
    group_load = np.zeros(num_groups)
    for idx in indices:
        min_group_idx = np.argmin(group_load)
        groups[min_group_idx].append(idx)
        if equal_size and len(groups[min_group_idx]) == group_size:
            group_load[min_group_idx] = float('inf')
        else:
            group_load[min_group_idx] += load[idx]
    return groups


def cycle(data_loader: DataLoader) -> Iterator:
    while True:
        for data in data_loader:
            if isinstance(data_loader.sampler, ResumableSampler):
                data_loader.sampler.idx += data_loader.batch_size   # type: ignore[attr-defined]
            yield data
        if isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.epoch += 1
        if isinstance(data_loader.sampler, ResumableSampler):
            data_loader.sampler.epoch += 1
            data_loader.sampler.idx = 0
        

class ResumableSampler(Sampler):
    """
    Distributed sampler that is resumable.

    Args:
        dataset: Dataset used for sampling.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
    """

    def __init__(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.epoch = 0
        self.idx = 0
        self.drop_last = drop_last
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.world_size != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.world_size) / self.world_size  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.world_size)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.world_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.world_size]
        
        # resume from previous state
        indices = indices[self.idx:]

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def state_dict(self) -> dict[str, int]:
        return {
            'epoch': self.epoch,
            'idx': self.idx,
        }
        
    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.idx = state_dict['idx']
        

class BalancedResumableSampler(ResumableSampler):
    """
    Distributed sampler that is resumable and balances the load among the processes.

    Args:
        dataset: Dataset used for sampling.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
    """

    def __init__(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        batch_size: int = 1,
    ) -> None:
        assert hasattr(dataset, 'loads'), 'Dataset must have "loads" attribute to use BalancedResumableSampler'
        super().__init__(dataset, shuffle, seed, drop_last)
        self.batch_size = batch_size
        self.loads = dataset.loads
        
    def __iter__(self) -> Iterator:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # balance load among processes
        num_batches = len(indices) // (self.batch_size * self.world_size)
        balanced_indices = []
        for i in range(num_batches):
            start_idx = i * self.batch_size * self.world_size
            end_idx = (i + 1) * self.batch_size * self.world_size
            batch_indices = indices[start_idx:end_idx]
            batch_loads = [self.loads[idx] for idx in batch_indices]
            groups = load_balanced_group_indices(batch_loads, self.world_size, equal_size=True)
            balanced_indices.extend([batch_indices[j] for j in groups[self.rank]])
        
        # resume from previous state
        indices = balanced_indices[self.idx:]

        return iter(indices)


class FourWayResumableSampler(ResumableSampler):
    """Homogeneous distributed batches with exact 25% mode allocation.

    Every global batch (all ranks together) comes from one mode. Modes cycle
    point/image/rough/both, and declared subtypes cycle independently inside
    their parent mode. Pools wrap and reshuffle deterministically as needed.
    """

    def __init__(self, dataset, shuffle=True, seed=0, drop_last=False, batch_size=1):
        self.batch_size = int(batch_size)
        super().__init__(dataset, shuffle=shuffle, seed=seed, drop_last=drop_last)

    def _active_modes(self):
        """Canonical order of modes that actually have training pools."""
        return tuple(
            mode for mode in ('point', 'image', 'rough', 'both')
            if self.dataset.mode_subtype_indices.get(mode)
        )

    def _layout_signature(self):
        """Describe the batching layout that makes a saved cursor valid."""
        base = tuple(
            (mode, subtype, len(indices))
            for mode in self._active_modes()
            for subtype, indices in sorted(self.dataset.mode_subtype_indices[mode].items())
        )
        sparse = getattr(self.dataset, 'sparse_mode_subtype_indices', {})
        sparse_sig = tuple(
            (mode, subtype, len(indices),
             len(getattr(self.dataset, 'dense_mode_subtype_indices', {}).get(mode, {}).get(subtype, [])))
            for mode, subtype_map in sparse.items()
            for subtype, indices in sorted(subtype_map.items())
        )
        return base + (('sparse_point_ratio', float(getattr(self.dataset, 'sparse_point_ratio', 0))),) + sparse_sig

    def state_dict(self):
        state = super().state_dict()
        state.update({
            'batch_size': self.batch_size,
            'world_size': self.world_size,
            'layout_signature': self._layout_signature(),
        })
        return state

    def load_state_dict(self, state_dict):
        self.epoch = state_dict.get('epoch', 0)
        compatible = (
            state_dict.get('batch_size') == self.batch_size
            and state_dict.get('world_size') == self.world_size
            and state_dict.get('layout_signature') == self._layout_signature()
        )
        saved_idx = int(state_dict.get('idx', 0))
        # A cursor from another batch size or data layout can start halfway
        # through a homogeneous four-way batch, which makes default_collate
        # combine incompatible dictionaries (for example point_cond with an
        # image-only item).  Preserve model/optimizer/step, but restart this
        # epoch's data order at a valid mode boundary.
        self.idx = saved_idx if compatible and saved_idx % self.batch_size == 0 else 0

    def __iter__(self):
        modes = self._active_modes()
        num_modes = len(modes)
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        pools = {}
        positions = {}
        for mode, subtype_map in self.dataset.mode_subtype_indices.items():
            for subtype, values in subtype_map.items():
                values = list(values)
                if self.shuffle:
                    order = torch.randperm(len(values), generator=generator).tolist()
                    values = [values[i] for i in order]
                pools[(mode, subtype)] = values
                positions[(mode, subtype)] = 0
        sparse_maps = getattr(self.dataset, 'sparse_mode_subtype_indices', {})
        dense_maps = getattr(self.dataset, 'dense_mode_subtype_indices', {})
        for mode, subtype_map in sparse_maps.items():
            for subtype, sparse_values in subtype_map.items():
                for density, source in (('sparse', sparse_values),
                                        ('dense', dense_maps.get(mode, {}).get(subtype, []))):
                    values = list(source)
                    if values and self.shuffle:
                        order = torch.randperm(len(values), generator=generator).tolist()
                        values = [values[i] for i in order]
                    pools[(mode, subtype, density)] = values
                    positions[(mode, subtype, density)] = 0
        for mode, source in getattr(self.dataset, 'sparse_mode_indices', {}).items():
            values = list(source)
            if values and self.shuffle:
                order = torch.randperm(len(values), generator=generator).tolist()
                values = [values[i] for i in order]
            pools[(mode, '__global_sparse')] = values
            positions[(mode, '__global_sparse')] = 0

        def take_from(key, count):
            pool = pools[key]
            selected = []
            while len(selected) < count:
                position = positions[key]
                take = min(count - len(selected), len(pool) - position)
                selected.extend(pool[position:position + take])
                position += take
                if position == len(pool):
                    position = 0
                    if self.shuffle:
                        order = torch.randperm(len(pool), generator=generator).tolist()
                        pool = [pool[i] for i in order]
                        pools[key] = pool
                positions[key] = position
            return selected

        global_batch = self.batch_size * self.world_size
        num_global_batches = max(num_modes, math.ceil(len(self.dataset) / global_batch / num_modes) * num_modes)
        local_indices = []
        mode_occurrences = {mode: 0 for mode in modes}
        stratum_occurrences = {}
        for batch_number in range(num_global_batches):
            mode = modes[batch_number % num_modes]
            subtypes = sorted(self.dataset.mode_subtype_indices[mode])
            subtype = subtypes[mode_occurrences[mode] % len(subtypes)]
            mode_occurrences[mode] += 1
            key = (mode, subtype)
            sparse_key, dense_key = (mode, '__global_sparse'), (mode, subtype, 'dense')
            ratio = float(getattr(self.dataset, 'sparse_point_ratio', 0))
            # The requested exposure target applies to the point-only branch.
            # Both-mode batches retain their established image/point pairing.
            if mode == 'point' and ratio > 0 and pools.get(sparse_key) and pools.get(dense_key):
                occurrence = stratum_occurrences.get(key, 0)
                sparse_count = (math.floor((occurrence + 1) * global_batch * ratio)
                                - math.floor(occurrence * global_batch * ratio))
                stratum_occurrences[key] = occurrence + 1
                selected = take_from(sparse_key, sparse_count)
                selected += take_from(dense_key, global_batch - sparse_count)
                if self.shuffle:
                    order = torch.randperm(len(selected), generator=generator).tolist()
                    selected = [selected[i] for i in order]
            else:
                selected = take_from(key, global_batch)
            start = self.rank * self.batch_size
            local_indices.extend(selected[start:start + self.batch_size])
        return iter(local_indices[self.idx:])

    def __len__(self):
        global_batch = self.batch_size * self.world_size
        num_modes = len(self._active_modes())
        return max(num_modes, math.ceil(len(self.dataset) / global_batch / num_modes) * num_modes) * self.batch_size
