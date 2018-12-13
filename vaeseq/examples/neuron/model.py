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

"""The model for multi-neuron spiking.

At each time step, we predict:
* K in [0, 5], indicating which finger (1-5) was pressed, or the null condition
  (0)

When generating, we emit the top finger per timestep. (Not yet implemented)
"""

from __future__ import print_function

import numpy as np
import tensorflow as tf

from vaeseq import codec as codec_mod
from vaeseq import context as context_mod
from vaeseq import model as model_mod
from vaeseq import util

from . import dataset as dataset_mod


class Model(model_mod.ModelBase):
    """Putting everything together."""

    def _make_encoder(self):
        """Constructs an encoder for a single observation."""
        return codec_mod.MLPObsEncoder(self.hparams, name="obs_encoder")

    def _make_decoder(self):
        """Constructs a decoder for a single observation.

        Implementation Options:
          1. We use a MLPObsDecoder to map from state to firing rate
          2. We use a categorical variable back-end for the finger class
             Note: possible that we should just be getting the firing rate out
        """
        # TODO: This should probably be a normal MLPObsDecoder
        # Not sure if CategoricalDecoder is applicable because it's not the
        # observed state, it's the data labels
        # Not sure if this should be more like text/model.py or midi/model.py
        return codec_mod.MLPObsDecoder(
            self.hparams,
            decoder=codec_mod.CategoricalDecoder(name="finger_decoder"),
            param_size=6,
            name="obs_decoder"
        )

    def _make_feedback(self):
        """Constructs the feedback Context.

        The context feeds back the observations from previous time steps as an
        input to the current RNN's time-step. i.e. Each RNN gets a time-window
        of observed neural data, not just the current point in time.
        """
        history_combiner = codec_mod.EncoderSequence(
            [codec_mod.FlattenEncoder(),
             util.make_mlp(self.hparams,
                           self.hparams.history_encoder_fc_layers)],
            name="history_combiner"
        )
        return context_mod.Accumulate(
            obs_encoder=self.encoder,
            history_size=self.hparams.history_size,
            history_combiner=history_combiner)

    def _make_dataset(self, files):
        dataset = dataset_mod.binned_spike_sequences(
            files,
            util.batch_size(self.hparams),
            util.sequence_size(self.hparams)
            )
        iterator = dataset.make_initializable_iterator()
        tf.add_to_collection(tf.GraphKeys.LOCAL_INIT_OP, iterator.initializer)
        single_trial_data = iterator.get_next()
        observed = (single_trial_data,)
        inputs = None
        return inputs, observed

    def _make_output_summary(self, tag, observed):
        """Minimal output summary.

        TODO: add some of the info from plot_lfads.py to this summary.
        and lfads.py:summarize_all
        """
        single_trial_spikes, = observed
        return tf.summary.scalar(
            tag + "/spike_avg",
            tf.reduce_mean(single_trial_spikes),
            )
