# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dummy (synthetic) data modules for profiling / benchmarking / OOM testing.

These substitute for the real ``DataModule`` / ``InferenceDataModule`` so the
production training and prediction paths can run without any on-disk dataset,
alignments, templates, or checkpoints. A single synthetic batch (built by
``openfold3.core.data.synthetic``) is yielded ``n_steps`` times.

DDP note: a map-style ``Dataset`` of length ``n_steps`` is used together with
``use_distributed_sampler=False`` (set by the dummy experiment runners). With
the distributed sampler disabled Lightning does not slice the dataset across
ranks, so every rank iterates the full ``n_steps`` batches -> DDP steps stay
synchronized (no all-reduce hang from uneven per-rank step counts). Each rank
sees identical batches, which is what we want for a benchmark.
"""

import logging

import pytorch_lightning as pl
from pydantic import BaseModel
from pydantic import ConfigDict as PydanticConfigDict
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from openfold3.core.data.synthetic import (
    build_inference_batch,
    build_training_batch,
)

logger = logging.getLogger(__name__)


class _EpochAwareSequentialSampler(SequentialSampler):
    """SequentialSampler with the ``epoch``/``set_epoch`` surface the production
    training hooks (``on_train_epoch_start``/``end``) read on the real
    ``OF3DistributedSampler``. Sequential order is fine: every rank iterates the
    full synthetic dataset (we disable the distributed sampler)."""

    epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class DummyDataConfig(BaseModel):
    """Configuration for synthetic ("dummy") input data.

    When ``enabled``, the training / prediction run is fed synthetic batches of
    the given shape instead of a real dataset. Used for profiling, benchmarking,
    and OOM validation across local / k8s / lucid without staging data.
    """

    model_config = PydanticConfigDict(extra="forbid")

    enabled: bool = False
    # Token (residue) count, i.e. the crop size.
    n_token: int = 384
    # MSA depth fed to the model (subsampled internally per recycle).
    n_msa: int = 1024
    n_templ: int = 4
    batch_size: int = 1
    # Total number of batches yielded (== optimizer/predict steps per rank).
    # Step timing is recorded per step (train_timing_*.jsonl for train, per-step
    # timing.json for predict); average over the post-warmup steps downstream.
    n_steps: int = 5
    # Training only: synthesize symmetry features so the real multi-chain
    # permutation alignment runs instead of the degenerate naive fallback.
    full_permutation: bool = True
    # Split tokens into multiple chains (multimer-like asym layout).
    is_multimer: bool = False


def _identity_collate(items):
    """The dataset element is already a fully collated batch (it carries its own
    batch dim), so with ``DataLoader(batch_size=1)`` we just unwrap the
    single-element list rather than letting the default collate add a second
    batch dim."""
    return items[0]


class _DummyBatchDataset(Dataset):
    """Yields the same pre-built batch ``n_steps`` times.

    ``__getitem__`` returns a shallow copy of the canonical batch dict so that
    per-step in-place top-level mutations (the model pops ``ref_space_uid_to_perm``
    during the training forward; ``predict_step`` reassigns ``batch['seed']``) do
    not corrupt the batch reused on the next step. Tensor values are shared (read
    only) and Lightning moves them to device into fresh tensors, leaving the CPU
    originals intact.
    """

    def __init__(
        self, batch: dict, n_steps: int, per_step_query_id: bool = False
    ) -> None:
        self._batch = batch
        self._n_steps = n_steps
        self._per_step_query_id = per_step_query_id

    def __len__(self) -> int:
        return self._n_steps

    def __getitem__(self, idx: int) -> dict:
        item = dict(self._batch)
        if self._per_step_query_id and "query_id" in item:
            # Give each step a distinct query_id so PredictTimer's per-query
            # timing.json files (output_dir/<query_id>/seed_<n>/) are one per
            # step instead of overwriting a single file.
            bs = len(item["query_id"])
            item["query_id"] = [
                f"dummy_s{idx}" if bs == 1 else f"dummy_s{idx}_b{b}"
                for b in range(bs)
            ]
        return item


class DummyDataModule(pl.LightningDataModule):
    """LightningDataModule serving synthetic batches for train or predict.

    Args:
        config: The :class:`DummyDataConfig`.
        mode: ``"train"`` or ``"predict"``. Selects the batch builder (training
            batches carry ground-truth + permutation features; prediction
            batches carry the metadata ``predict_step`` reads).
    """

    def __init__(self, config: DummyDataConfig, mode: str) -> None:
        super().__init__()
        if mode not in ("train", "predict"):
            raise ValueError(f"mode must be 'train' or 'predict', got {mode!r}")
        self.config = config
        self.mode = mode
        self._batch: dict | None = None
        # Read by the production training epoch hooks; empty for synthetic data
        # (no in-order dataset sampling to track).
        self.next_dataset_indices: dict = {}

    def prepare_data(self) -> None:
        # No MSA server / template preprocessing / query-set broadcast.
        return None

    def setup(self, stage=None) -> None:
        c = self.config
        if self._batch is not None:
            return
        if self.mode == "train":
            self._batch = build_training_batch(
                n_token=c.n_token,
                n_msa=c.n_msa,
                n_templ=c.n_templ,
                batch_size=c.batch_size,
                full_permutation=c.full_permutation,
                is_multimer=c.is_multimer,
            )
        else:
            self._batch = build_inference_batch(
                n_token=c.n_token,
                n_msa=c.n_msa,
                n_templ=c.n_templ,
                batch_size=c.batch_size,
                is_multimer=c.is_multimer,
            )
        logger.info(
            "DummyDataModule(%s): n_token=%d n_msa=%d n_templ=%d batch_size=%d "
            "n_steps=%d",
            self.mode, c.n_token, c.n_msa, c.n_templ, c.batch_size, c.n_steps,
        )

    def _loader(self, epoch_aware_sampler: bool = False) -> DataLoader:
        if self._batch is None:
            self.setup()
        dataset = _DummyBatchDataset(
            self._batch,
            self.config.n_steps,
            per_step_query_id=(self.mode == "predict"),
        )
        # Provide the epoch-aware sampler for training so the production epoch
        # hooks (which read sampler.epoch) work without the OF3DistributedSampler.
        sampler = _EpochAwareSequentialSampler(dataset) if epoch_aware_sampler else None
        # num_workers=0: tensors are pre-materialized, so workers add no value and
        # this sidesteps DataLoader shared-memory entirely.
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=_identity_collate,
            sampler=sampler,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(epoch_aware_sampler=True)

    def predict_dataloader(self) -> DataLoader:
        return self._loader()
