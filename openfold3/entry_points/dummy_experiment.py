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

"""Experiment runners that feed synthetic ("dummy") data through the production
training / prediction paths.

These subclass the production runners and override only the data-source and
callback seams, so timing / memory-snapshot / torch-profiler numbers come from
exactly the same ``pl.Trainer`` configuration, precision plugin, strategy, and
callbacks as a real run. They are selected by ``run_openfold train``/``predict``
when ``dummy_data.enabled`` is set in the runner yaml.
"""

import logging
from functools import cached_property

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from openfold3.core.data.framework.dummy_data import DummyDataModule
from openfold3.core.runners.writer import OF3OutputWriter
from openfold3.core.utils.callbacks import LogInferenceQuerySet
from openfold3.entry_points.experiment_runner import (
    InferenceExperimentRunner,
    TrainingExperimentRunner,
)

logger = logging.getLogger(__name__)


class DummyTrainingExperimentRunner(TrainingExperimentRunner):
    """Training on synthetic data via the production fit path."""

    def __init__(self, experiment_config) -> None:
        super().__init__(experiment_config)
        self.dummy_config = experiment_config.dummy_data
        # Disable the distributed sampler so each rank iterates the full
        # n_steps (synchronized DDP, no auto-injected DistributedSampler).
        # Set here (post config validation) rather than in the yaml so the
        # stale validate_distributed_sampler_settings warning never fires.
        self.pl_trainer_args.use_distributed_sampler = False
        # One pass over the n_steps synthetic batches; no validation; no
        # checkpoint writes (this is a profiling / benchmarking run).
        self.pl_trainer_args.max_epochs = 1
        self.pl_trainer_args.num_sanity_val_steps = 0
        self.pl_trainer_args.limit_val_batches = 0
        self.pl_trainer_args.enable_checkpointing = False
        # Turn on per-step training timing (production PredictTimer → step time
        # logged + written to train_timing_rank*.jsonl under output_dir).
        self.model_config.settings.debug.log_iteration_time = True

    @cached_property
    def lightning_data_module(self) -> pl.LightningDataModule:
        return DummyDataModule(self.dummy_config, mode="train")

    @cached_property
    def callbacks(self):
        # Drop ModelCheckpoint: enable_checkpointing is False for dummy runs, and
        # Lightning errors if a ModelCheckpoint is present while it is disabled.
        # Timing (PredictTimer) and peak memory (MemorySnapshot) come from the
        # standard production callbacks, not a dummy-specific one.
        return [c for c in super().callbacks if not isinstance(c, ModelCheckpoint)]


class DummyInferenceExperimentRunner(InferenceExperimentRunner):
    """Prediction on synthetic data via the production predict path."""

    def __init__(self, experiment_config, *args, **kwargs) -> None:
        super().__init__(experiment_config, *args, **kwargs)
        self.dummy_config = experiment_config.dummy_data
        self.pl_trainer_args.use_distributed_sampler = False

    @cached_property
    def lightning_module(self) -> pl.LightningModule:
        """Random-initialized weights (no skip_random_init): with no checkpoint
        loaded we want sensibly-initialized parameters for a stable forward."""
        return self.project_entry.runner(self.model_config, log_dir=self.log_dir)

    @cached_property
    def lightning_data_module(self) -> pl.LightningDataModule:
        return DummyDataModule(self.dummy_config, mode="predict")

    @cached_property
    def callbacks(self):
        # Drop the callbacks that require real query metadata / structures:
        # OF3OutputWriter (writes mmCIF from batch['atom_array']) and
        # LogInferenceQuerySet (reads datamodule.inference_config.query_set).
        # PredictTimer is KEPT: it writes the canonical per-query timing.json
        # (runtime_s) that inference tooling reads. The dummy data module gives
        # each step a unique query_id so the per-step timings don't overwrite,
        # and PredictTimer creates its own output subdir. Peak memory comes
        # from MemorySnapshot (production callback).
        return [
            c
            for c in super().callbacks
            if not isinstance(c, (OF3OutputWriter, LogInferenceQuerySet))
        ]

    def setup(self) -> None:
        """Set up environment. Skip checkpoint loading when none is configured
        (dummy runs default to random init)."""
        if self.ckpt_path is None:
            # Mirror InferenceExperimentRunner.setup minus the checkpoint load.
            super(InferenceExperimentRunner, self).setup()
            self._log_experiment_config()
            self._log_model_config()
            logger.info(
                "Dummy inference: using random-initialized weights "
                "(no checkpoint configured)."
            )
        else:
            super().setup()
        # Seed BEFORE the model (random init) and the synthetic batch are built
        # (both happen lazily at trainer.predict in run()). Otherwise the random
        # weights and the synthetic atom composition come from the unseeded
        # global RNG and vary per process, so two runs are not comparable. The
        # in-loop predict_step reseed only fixes the per-step diffusion noise,
        # which happens after the batch is already built.
        # InferenceExperimentSettings exposes `seeds` (int | list[int]); the base
        # runner normalizes it onto self.seeds. Use the first seed.
        seed = self.seeds[0] if isinstance(self.seeds, list) else self.seeds
        pl.seed_everything(seed, workers=True)
        logger.info("Dummy inference: seeded weights + synthetic data with %d", seed)

    def run(self, *_args, **_kwargs) -> None:
        """Run prediction on the synthetic data module.

        Ignores any inference_query_set argument; there is no query set for a
        dummy run.
        """
        logger.info("Beginning dummy inference prediction")
        self.trainer.predict(
            model=self.lightning_module,
            datamodule=self.lightning_data_module,
            return_predictions=False,
        )
