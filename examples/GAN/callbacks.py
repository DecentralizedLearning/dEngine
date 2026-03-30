from __future__ import annotations

import logging

from npy_append_array import NpyAppendArray
import numpy as np

from dengine.callbacks import PeriodicCallback
from dengine.config.constants import METRICS_DIR_NAME
from dengine.models.memnet import MemNet
from dengine.callbacks.decorators import register_callback


@register_callback()
class GANCallback(PeriodicCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_path_root = self.experiment_configuration.output_directory
        self._samples_outfolder = (
            (output_path_root / 'samples') / self.client.UUID
        )
        self._samples_outfolder.mkdir(parents=True, exist_ok=True)
        self._loss_outfolder = (
            (output_path_root / METRICS_DIR_NAME) / self.client.UUID
        )

        self._G_outfile = self._loss_outfolder / 'round_G_loss.npy'
        self._G_outfile.parent.mkdir(parents=True, exist_ok=True)

        self._D_outfile = self._loss_outfolder / 'round_D_loss.npy'
        self._D_outfile.parent.mkdir(parents=True, exist_ok=True)

    def _dump_loss(
        self,
        g_loss: float,
        d_loss: float,
        *args, **kwargs
    ):
        with NpyAppendArray(self._G_outfile) as tr_out_file_array:
            tr_out_file_array.append(np.array([g_loss]))
        logging.info(f'Appended trainining loss {g_loss} to: {self._G_outfile}')

        with NpyAppendArray(self._D_outfile) as valid_out_file_array:
            valid_out_file_array.append(np.array([d_loss]))
        logging.info(f'Appended validation loss {d_loss} to: {self._D_outfile}')

    def _dump_samples(self, *args, **kwargs):
        assert isinstance(self.client.model, MemNet)
        for i, interpolation in enumerate(self.client.model.sample_validation_batch(50)):
            fname = f"from={interpolation.source_label_idx},to={interpolation.target_label_idx},{i}.gif"
            interpolation.frames[0].save(
                self._samples_outfolder / fname,
                save_all=True,
                append_images=interpolation.frames[1:],
                duration=100,
                loop=1
            )
        logging.info(f"GIFs dumped at: {self._samples_outfolder}")

    def training_epoch_end(
        self,
        epoch: int,
        g_loss: float,
        d_loss: float,
        *args, **kwargs
    ):
        self._dump_loss(g_loss=g_loss, d_loss=d_loss)
        self._dump_samples()
