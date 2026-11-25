"""Batch sampler with simple case/control balancing and optional natural batches."""

from __future__ import annotations

import logging
import random
from typing import Iterator, List, Protocol

from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class SurvivalStratumDataset(Protocol):
    n_years: int

    def __len__(self) -> int: ...

    def survival_stratum_for_index(self, idx: int) -> tuple[bool, int]: ...


class HorizonBalancedBatchSampler(Sampler[List[int]]):
    """
    Mostly case/control balanced batches (~50% cancer / ~50% controls), mixed with
    occasional natural batches to reduce oversampling memorization of rare cases.

    Balanced batch (default 4 of every 5):
      - half the batch: random cancer cases (without replacement per epoch pool)
      - half the batch: random controls (without replacement per epoch pool)
      - horizon distribution within each half follows dataset prevalence

    Natural batch (every ``random_batch_every``-th batch, e.g. 5th of 5):
      - ``batch_size`` uniform random indices from the full train set
      - reflects natural class prevalence (~7% cases for CSAW pairs)
    """

    def __init__(
        self,
        dataset: SurvivalStratumDataset,
        batch_size: int,
        seed: int = 0,
        random_batch_every: int = 5,
    ) -> None:
        if batch_size < 2:
            raise ValueError("batch_size must be >= 2 for case/control balancing")
        if random_batch_every is not None and random_batch_every < 2:
            raise ValueError("random_batch_every must be >= 2 (or 0 to disable natural batches)")

        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.random_batch_every = random_batch_every or 0
        self._epoch = 0
        self._rng = random.Random(seed)

        self._all_pos: list[int] = []
        self._all_neg: list[int] = []

        for idx in range(len(dataset)):
            is_event, _bucket = dataset.survival_stratum_for_index(idx)
            if is_event:
                self._all_pos.append(idx)
            else:
                self._all_neg.append(idx)

        self._all_indices = list(range(len(dataset)))

        if not self._all_pos:
            raise ValueError("HorizonBalancedBatchSampler: no cancer cases found in the dataset")
        if not self._all_neg:
            raise ValueError("HorizonBalancedBatchSampler: no controls found in the dataset")

        n_balanced = max(0, self.random_batch_every - 1) if self.random_batch_every else 1
        n_natural = 1 if self.random_batch_every else 0
        cycle = n_balanced + n_natural
        effective_pos_rate = (
            (n_balanced * 0.5 + n_natural * (len(self._all_pos) / len(dataset))) / cycle
            if cycle
            else 0.5
        )

        logger.info(
            "HorizonBalancedBatchSampler: %d samples | pos=%d (%.1f%%) neg=%d | batch_size=%d",
            len(dataset),
            len(self._all_pos),
            100.0 * len(self._all_pos) / len(dataset),
            len(self._all_neg),
            batch_size,
        )
        logger.info(
            "  balanced batch: %d cases + %d controls (random, dataset horizon mix)",
            batch_size // 2,
            batch_size - batch_size // 2,
        )
        if self.random_batch_every:
            logger.info(
                "  mix: 1 natural batch every %d (effective ~%.1f%% cases per batch)",
                self.random_batch_every,
                100.0 * effective_pos_rate,
            )
        else:
            logger.info("  mix: all batches balanced (~50%% cases)")

    def __len__(self) -> int:
        return max(1, len(self.dataset) // self.batch_size)

    def _shuffled_queue(self, pool: list[int]) -> list[int]:
        queue = list(pool)
        self._rng.shuffle(queue)
        return queue

    def _start_epoch_queues(self) -> None:
        self._pos_queue = self._shuffled_queue(self._all_pos)
        self._neg_queue = self._shuffled_queue(self._all_neg)

    def _pop_from_queue(self, queue: list[int], pool: list[int]) -> int:
        if not queue:
            queue.extend(self._shuffled_queue(pool))
        return queue.pop()

    def _natural_batch(self) -> list[int]:
        return self._rng.sample(self._all_indices, k=self.batch_size)

    def _balanced_batch(self) -> list[int]:
        pos_slots = self.batch_size // 2
        neg_slots = self.batch_size - pos_slots
        batch: list[int] = []

        for _ in range(pos_slots):
            batch.append(self._pop_from_queue(self._pos_queue, self._all_pos))
        for _ in range(neg_slots):
            batch.append(self._pop_from_queue(self._neg_queue, self._all_neg))

        self._rng.shuffle(batch)
        return batch

    def __iter__(self) -> Iterator[List[int]]:
        self._start_epoch_queues()

        for batch_idx in range(len(self)):
            use_natural = (
                self.random_batch_every > 0
                and (batch_idx + 1) % self.random_batch_every == 0
            )
            if use_natural:
                yield self._natural_batch()
            else:
                yield self._balanced_batch()

    def set_epoch(self, epoch: int) -> None:
        """Reseed per epoch for reproducibility (call from training loop if needed)."""
        self._epoch = epoch
        self._rng.seed(self.seed + epoch)
