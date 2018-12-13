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

"""Tests for dataset.py functionality."""

import os.path
import numpy as np
import tensorflow as tf

from vaeseq.examples.neuron import dataset as dataset_mod


class DatasetTest(tf.test.TestCase):
    _NUM_TIME_POINTS = 100

    def _write_spikes(self, file_idx):
        """Write a temporary HDF5 file with 1-trial of spiking data."""
        # temp_path = os.path.join(r'C:\Users\hst\Documents\cguan\vae-seq\build',
        temp_path = os.path.join(self.get_temp_dir(),
            "spikes_{}.h5".format(file_idx))
        dataset_mod.write_poisson_spikes(temp_path, self._NUM_TIME_POINTS)
        return temp_path

    def test_binned_spike_sequences(self):
        data_filenames = [self._write_spikes(3), self._write_spikes(8)]
        batch_size = 3
        sequence_size = self._NUM_TIME_POINTS
        dataset = dataset_mod.binned_spike_sequences(
            data_filenames, batch_size, sequence_size)
        iterator = dataset.make_initializable_iterator()
        batch = iterator.get_next()
        with self.test_session() as sess:
            sess.run(iterator.initializer)
            batch = sess.run(batch)
            self.assertAllEqual(batch.shape, [batch_size, sequence_size, dataset_mod._NUM_CHANNELS])


if __name__ == "__main__":
    tf.test.main()
