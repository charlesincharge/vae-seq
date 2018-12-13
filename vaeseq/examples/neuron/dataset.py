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
import tensorflow as tf

_SPIKE_FIELD_NAME = 'observed_spikes'

def binned_spike_sequences(filenames, batch_size, sequence_size, num_channels):
    """Returns a dataset of binned neural spike sequences from the given
    files.

    Args:
      filenames (list(str)): List of names of neural data files, formatted as
        binned spike counts
      batch_size (int): number of trials
      sequence_size (int): number of time bins of spiked data per trial
      num_channels (int): number of parallel recording channels per sequence
    """

    def _to_binned_spikes(data_fname, sequence_size, num_channels):
        """Load a file and return consecutive binned spike sequences.
        Skips incorrectly formatted files.

        Args:
          data_fname (str): HDF5 file containing binned spike data
            Note: it would not be difficult to support .mat  files, too.
            See scipy.io.loadmat/savemat

        Returns:
          binned_spikes (3-D int32 np.array): binned spiking data, each
            element indicating number of spikes in that time window
        """
        data_dict = {}
        try:
            with h5py.File(data_fname, 'r') as data_file:
                data_dict = {key : np.array(val) for key, val in list(data_file.items())}
        except (ValueError, FileNotFoundError) as exc:
            print("Skipping file: {0}. Error: {1}".format(data_fname, exc))
            return np.zeros([0, sequence_size, num_channels], dtype=np.int32)
        # Open issue in tensorflow-1.11.0. We should cast this to np.uint32, but
        # tf.batch doesn't work with uint32, only int32. Fixed in 1.12.0
        # https://github.com/tensorflow/tensorflow/issues/18586
        binned_spikes = data_dict[_SPIKE_FIELD_NAME].astype(np.int32)
        assert binned_spikes.shape[1] == sequence_size, \
            "expected: {}, actual: {}".format(sequence_size, binned_spikes.shape[1])
        assert binned_spikes.shape[2] == num_channels, \
            "expected: {}, actual: {}".format(num_channels, binned_spikes.shape[2])
        # TODO: change MATLAB format to (trial #, time, channel)
        # Although (trial #, channel, time) makes more sense to me in terms of dimensions
        return binned_spikes


    def _to_binned_spikes_dataset(data_fname):
        """Convert file to dataset of binned spikes.

        Args:
          data_fname (str): HDF5 or MATLAB file containing binned spiking data
        """
        binned_spikes, = tf.py_func(_to_binned_spikes,
                                   [data_fname, sequence_size, num_channels],
                                   [tf.int32])
        binned_spikes.set_shape([None, None, num_channels])
        return tf.data.Dataset.from_tensor_slices(binned_spikes)

    batch_size = tf.to_int64(batch_size)
    return (tf.data.Dataset.from_tensor_slices(filenames)
            .interleave(_to_binned_spikes_dataset,
                        cycle_length=batch_size * 5,
                        block_length=1)
            .repeat()
            .shuffle(1000)
            .batch(batch_size))


def write_poisson_spikes(path, num_time_points, num_channels):
    """Generate an arbitrary spiking sequence with a Poisson process.
    Write data to the path (Not yet implemented)
    Hard-coded as a testing utility, but could be generalized.
    Inspired by LFADS/synth_data/synthetic_data_utils.py:spikify_data
    Ranges from 1-10Hz neurons
    Args:
      num_channels (int): number of parallel recording channels per sequence
    """
    spikes = np.zeros([1, num_time_points, num_channels]).astype(np.int32)
    for channel in range(num_channels):
        spikes[0,:,channel] = np.random.poisson(channel, size=num_time_points)

    try:
        with h5py.File(path, 'a') as h5_file:
            h5_file.create_dataset(_SPIKE_FIELD_NAME, data=spikes)
    except IOError:
        print("Cannot open {} for writing".format(path))
        raise
