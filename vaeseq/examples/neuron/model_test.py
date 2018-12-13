# Copyright 2018 Google, Inc.,
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

"""Tests for training and generating Tensorflow summaries."""

import os.path
import tensorflow as tf
from vaeseq import model_test

from vaeseq.examples.neuron import dataset as dataset_mod
from vaeseq.examples.neuron import hparams as hparams_mod
from vaeseq.examples.neuron import model as model_mod


class ModelTest(model_test.ModelTest):

    def _write_neural_data(self, file_idx, num_time_bins, num_channels):
        """Write a temporary neural spiking data file.

        Args:
            num_time_bins (int): number of bins of "observed" spikes to generate
        """
        temp_path = os.path.join(self.get_temp_dir(),
                                 "spikes_{}.h5".format(file_idx))
        dataset_mod.write_poisson_spikes(temp_path, num_time_bins, num_channels)
        return temp_path

    def _setup_model(self, session_params):
        self.hparams = hparams_mod.make_hparams(
            rnn_hidden_sizes=[4, 4],
            obs_encoder_fc_layers=[32, 16],
            obs_decoder_fc_hidden_layers=[32],
            latent_decoder_fc_layers=[32],
            check_numerics=True)
        num_time_bins = self.hparams.sequence_size
        num_channels = self.hparams.num_recording_channels
        self.train_dataset = [self._write_neural_data(3, num_time_bins, num_channels),
                              self._write_neural_data(5, num_time_bins, num_channels)]
        self.valid_dataset = [self._write_neural_data(7, num_time_bins, num_channels),
                              self._write_neural_data(10, num_time_bins, num_channels)]
        self.model = model_mod.Model(self.hparams, session_params)


if __name__ == "__main__":
    tf.test.main()
