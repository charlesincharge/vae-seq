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

"""Model neural spiking sequences, recorded from multielectrode arrays."""

import argparse
import itertools
import os.path
import sys

import tensorflow as tf

from vaeseq.examples.neuron import hparams as hparams_mod
from vaeseq.examples.neuron import model as model_mod


def train(flags):
    model = model_mod.Model(
        hparams=hparams_mod.make_hparams(flags.hparams),
        session_params=flags)
    model.train(flags.train_files, flags.num_steps,
                valid_dataset=flags.valid_files)


def evaluate(flags):
    model = model_mod.Model(
        hparams=hparams_mod.make_hparams(flags.hparams),
        session_params=flags)
    model.evaluate(flags.eval_files, flags.num_steps)


def generate(flags):
    hparams = hparams_mod.make_hparams(flags.hparams)
    hparams.sequence_size = int(hparams.rate * flags.length)
    model = model_mod.Model(hparams=hparams, session_params=flags)
    raise NotImplementedError


# Argument parsing code below.

def common_args(args):
    model_mod.Model.SessionParams.add_parser_arguments(args)
    args.add_argument(
        "--hparams", default="",
        help="Model hyperparameter overrides.")


# TODO: annotate the data format of training-files
def train_args(args):
    common_args(args)
    args.add_argument(
        "--train-files", nargs="+",
        help="Neural spiking data to train on.",
        required=True)
    args.add_argument(
        "--valid-files", nargs="+",
        help="Neural spiking data to evaluate while training.")
    args.add_argument(
        "--num-steps", type=int, default=int(1e6),
        help="Number of training iterations.")
    args.set_defaults(entry=train)


def eval_args(args):
    common_args(args)
    args.add_argument(
        "--eval-files", nargs="+",
        help="Neural spiking data to evaluate.",
        required=True)
    args.add_argument(
        "--num-steps", type=int, default=int(1e3),
        help="Number of eval iterations.")
    args.set_defaults(entry=evaluate)


def generate_args(args):
    common_args(args)
    args.add_argument(
        "--out-dir",
        help="Where to store the generated sequences.",
        required=True)
    args.add_argument(
        "--length", type=float, default=1.,
        help="Length of the generated sequences, in seconds.")
    args.add_argument(
        "--num-samples", type=int, default=5,
        help="Number of sequences to generate.")
    args.set_defaults(entry=generate)


def main():
    args = argparse.ArgumentParser()
    subcommands = args.add_subparsers(title="subcommands")
    train_args(subcommands.add_parser(
        "train", help="Train a model."))
    eval_args(subcommands.add_parser(
        "evaluate", help="Evaluate a trained model."))
    generate_args(subcommands.add_parser(
        "generate", help="(Not yet implemented) Generate neural firing rates."))
    flags, unparsed_args = args.parse_known_args(sys.argv[1:])
    if not hasattr(flags, "entry"):
        args.print_help()
        return 1
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=lambda _unused_argv: flags.entry(flags),
               argv=[sys.argv[0]] + unparsed_args)


if __name__ == "__main__":
    main()
