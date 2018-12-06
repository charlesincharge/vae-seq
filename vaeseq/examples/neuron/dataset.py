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

"""Dataset for iterating over neural spiking recording files."""

from __future__ import print_function

import h5py
import numpy as np
import scipy.io
import tensorflow as tf

# TODO: remove hard-coding
_NUM_CHANNELS = 75
_SPIKE_FIELD_NAME = 'observed_spikes'

def binned_spike_sequences(filenames, batch_size, sequence_size, rate=100):
    """Returns a dataset of binned neural spike sequences from the given
    files."""

    def _to_binned_spikes(filename, sequence_size):
        """Load a file and return consecutive binned spike sequences.
        Skips incorrectly formatted files.

        Args:
          filename (str): HDF5 or MATLAB file containing binned spiking data

        Returns:
          binned_spikes (3-D tf.uint32 np.array): binned spiking data, each
            element indicating number of spikes in that time window
        """
        try:
            file_data = scipy.io.loadmat(tf.compat.as_text(filename))
        except (ValueError, FileNotFoundError) as exc:
            print("Skipping file: {0}. Error: {1}".format(filename, exc))
            return np.zeros(0, sequence_size, _NUM_CHANNELS)
        binned_spikes = file_data[_SPIKE_FIELD_NAME]
        assert binned_spikes.shape[1] == sequence_size, \
            "expected: {}, actual: {}".format(sequence_size, binned_spikes.shape[1])
        assert binned_spikes.shape[2] == _NUM_CHANNELS, \
            "expected: {}, actual: {}".format(_NUM_CHANNELS, binned_spikes.shape[0])
        # TODO: change MATLAB format to (trial #, time, channel)
        # Although (trial #, channel, time) makes more sense to me in terms of dimensions
        return binned_spikes


    def _to_binned_spikes_dataset(filename):
        """Convert file to dataset of binned spikes.

        Args:
          filename (str): HDF5 or MATLAB file containing binned spiking data
        """
        binned_spikes, = tf.py_func(_to_binned_spikes,
                                   [filename, sequence_size],
                                   [tf.uint32])
        binned_spikes.set_shape([None, None, _NUM_CHANNELS])
        return tf.data.Dataset.from_tensor_slices(binned_spikes)

    batch_size = tf.to_int64(batch_size)
    return (tf.data.Dataset.from_tensor_slices(filenames)
            .interleave(_to_binned_spikes_dataset,
                        cycle_length=batch_size * 5,
                        block_length=1)
            .repeat()
            .shuffle(1000)
            .batch(batch_size))


def write_poisson_spikes(path, num_time_points):
    """Generate an arbitrary spiking sequence with a Poisson process.
    Write data to the path (Not yet implemented)
    Hard-coded as a testing utility, but could be generalized.
    Inspired by LFADS/synth_data/synthetic_data_utils.py:spikify_data
    Ranges from 1-10Hz neurons
    Args:
    """
    spikes = np.zeros([1, _NUM_CHANNELS, num_time_points]).astype(np.int)
    for channel in range(_NUM_CHANNELS):
        spikes[0, channel,:] = np.random.poisson(channel, size=num_time_points)

    try:
        with h5py.File(path, 'a') as h5_file:
            h5_file.create_dataset(_SPIKE_FIELD_NAME, data=spikes)
    except IOError:
        print("Cannot open {} for writing".format(path))
        raise
