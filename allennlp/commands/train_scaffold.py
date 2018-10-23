"""
The ``train`` subcommand can be used to train a model.
It requires a configuration file and a directory in
which to write the results.

.. code-block:: bash

   $ python -m allennlp.run train --help
   usage: run [command] train [-h] -s SERIALIZATION_DIR param_path

   Train the specified model on the specified dataset.

   positional arguments:
   param_path            path to parameter file describing the model to be trained

   optional arguments:
    -h, --help            show this help message and exit
    -s SERIALIZATION_DIR, --serialization-dir SERIALIZATION_DIR
                            directory in which to save the model and its logs
"""
from typing import Dict, Iterable
import argparse
import logging
import os
import random
import re

import torch

from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.commands.evaluate import evaluate
from allennlp.commands.subcommand import Subcommand
from allennlp.commands.train import train_model_from_args, train_model_from_file, \
                                    create_serialization_dir
from allennlp.common.params import Params
from allennlp.common.util import prepare_environment, prepare_global_logging, \
                                 get_frozen_and_tunable_parameter_names, dump_metrics
from allennlp.data import Vocabulary
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.archival import archive_model, CONFIG_NAME
from allennlp.models.model import Model, _DEFAULT_WEIGHTS
from allennlp.training.scaffolded_trainer import ScaffoldedTrainer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TrainScaffold(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Train the specified model on the specified dataset.'''
        subparser = parser.add_parser(name, description=description, help='Train a model')

        subparser.add_argument('param_path',
                               type=str,
                               help='path to parameter file describing the model to be trained')

        subparser.add_argument('-s', '--serialization-dir',
                               required=True,
                               type=str,
                               help='directory in which to save the model and its logs')

        subparser.add_argument('-r', '--recover',
                               action='store_true',
                               default=False,
                               help='recover training from the state in serialization_dir')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a JSON structure used to override the experiment configuration')

        subparser.add_argument('--file-friendly-logging',
                               action='store_true',
                               default=False,
                               help='outputs tqdm status on separate lines and slows tqdm refresh rate')

        subparser.set_defaults(func=train_model_from_args)

        return subparser


def datasets_from_params(params: Params) -> Dict[str, Iterable[Instance]]:
    """
    Load all the datasets specified by the config.
    """
    dataset_reader = DatasetReader.from_params(params.pop('dataset_reader'))
    validation_dataset_reader_params = params.pop("validation_dataset_reader", None)

    validation_and_test_dataset_reader: DatasetReader = dataset_reader
    if validation_dataset_reader_params is not None:
        logger.info("Using a separate dataset reader to load validation and test data.")
        validation_and_test_dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)

    train_data_path = params.pop('train_data_path')
    logger.info("Reading training data from %s", train_data_path)
    train_data = dataset_reader.read(train_data_path)

    datasets: Dict[str, Iterable[Instance]] = {"train": train_data}

    # 2. Auxiliary train data
    aux_dataset_reader = DatasetReader.from_params(params.pop('aux_dataset_reader'))
    aux_train_data_path = params.pop('aux_train_data_path')
    logger.info("Reading AUX training data from %s", aux_train_data_path)
    aux_train_data = aux_dataset_reader.read(aux_train_data_path)

    # Adjust auxiliary train, if it is smaller than primary train.
    aux_train_size = len(aux_train_data.instances)
    if aux_train_size < len(train_data.instances):
        difference = len(train_data.instances) - aux_train_size
        aux_sample = [random.choice(aux_train_data.instances) for _ in range(difference)]
        aux_train_data = Iterable(aux_train_data.instances + aux_sample)
        logger.info("Inflating auxiliary train data from %d to %d samples",
                    aux_train_size, len(aux_train_data.instances))
        datasets["aux_train"] = aux_train_data

    validation_data_path = params.pop('validation_data_path', None)
    if validation_data_path is not None:
        logger.info("Reading validation data from %s", validation_data_path)
        validation_data = validation_and_test_dataset_reader.read(validation_data_path)
        datasets["validation"] = validation_data

    test_data_path = params.pop("test_data_path", None)
    if test_data_path is not None:
        logger.info("Reading test data from %s", test_data_path)
        test_data = validation_and_test_dataset_reader.read(test_data_path)
        datasets["test"] = test_data

    return datasets


def train_model(params: Params,
                serialization_dir: str,
                file_friendly_logging: bool = False,
                recover: bool = False) -> Model:
    """
    Trains the model specified in the given :class:`Params` object, using the data and training
    parameters also specified in that object, and saves the results in ``serialization_dir``.

    Parameters
    ----------
    params : ``Params``
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir : ``str``
        The directory in which to save results and logs.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we add newlines to tqdm output, even on an interactive terminal, and we slow
        down tqdm's output to only once every 10 seconds.
    recover : ``bool``, optional (default=False)
        If ``True``, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see the ``fine-tune`` command.

    Returns
    -------
    best_model: ``Model``
        The model with the best epoch weights.
    """
    prepare_environment(params)

    create_serialization_dir(params, serialization_dir, recover)
    prepare_global_logging(serialization_dir, file_friendly_logging)

    check_for_gpu(params.get('trainer').get('cuda_device', -1))

    params.to_file(os.path.join(serialization_dir, CONFIG_NAME))

    all_datasets = datasets_from_params(params)
    datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", all_datasets))

    for dataset in datasets_for_vocab_creation:
        if dataset not in all_datasets:
            raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {dataset}")

    logger.info("From dataset instances, %s will be considered for vocabulary creation.",
                ", ".join(datasets_for_vocab_creation))
    vocab = Vocabulary.from_params(
            params.pop("vocabulary", {}),
            (instance for key, dataset in all_datasets.items()
             for instance in dataset
             if key in datasets_for_vocab_creation)
    )
    vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

    model = Model.from_params(vocab=vocab, params=params.pop('model'))

    # Initializing the model can have side effect of expanding the vocabulary
    vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

    iterator = DataIterator.from_params(params.pop("iterator"))
    iterator.index_with(vocab)
    validation_iterator_params = params.pop("validation_iterator", None)
    if validation_iterator_params:
        validation_iterator = DataIterator.from_params(validation_iterator_params)
        validation_iterator.index_with(vocab)
    else:
        validation_iterator = None

    aux_iterator = DataIterator.from_params(params.pop("aux_iterator"))
    aux_iterator.index_with(vocab)

    train_data = all_datasets['train']
    aux_train_data = all_datasets.get('aux_train')
    validation_data = all_datasets.get('validation')
    test_data = all_datasets.get('test')

    trainer_params = params.pop("trainer")
    no_grad_regexes = trainer_params.pop("no_grad", ())
    for name, parameter in model.named_parameters():
        if any(re.search(regex, name) for regex in no_grad_regexes):
            parameter.requires_grad_(False)

    frozen_parameter_names, tunable_parameter_names = \
                   get_frozen_and_tunable_parameter_names(model)
    logger.info("Following parameters are Frozen  (without gradient):")
    for name in frozen_parameter_names:
        logger.info(name)
    logger.info("Following parameters are Tunable (with gradient):")
    for name in tunable_parameter_names:
        logger.info(name)

    trainer_params = params.pop("trainer")
    trainer = ScaffoldTrainer.from_params(model=model,
                                          serialization_dir=serialization_dir,
                                          iterator=iterator,
                                          aux_iterator=aux_iterator,
                                          train_dataset=train_data,
                                          aux_train_dataset=aux_train_data,
                                          validation_dataset=validation_data,
                                          params=trainer_params,
                                          files_to_archive=params.files_to_archive)

    evaluate_on_test = params.pop_bool("evaluate_on_test", False)
    params.assert_empty('base train command')

    try:
        metrics = trainer.train()
    except KeyboardInterrupt:
        # if we have completed an epoch, try to create a model archive.
        if os.path.exists(os.path.join(serialization_dir, _DEFAULT_WEIGHTS)):
            logging.info("Training interrupted by the user. Attempting to create "
                         "a model archive using the current best epoch weights.")
            archive_model(serialization_dir, files_to_archive=params.files_to_archive)
        raise

    # Now tar up results
    archive_model(serialization_dir, files_to_archive=params.files_to_archive)

    logger.info("Loading the best epoch weights.")
    best_model_state_path = os.path.join(serialization_dir, 'best.th')
    best_model_state = torch.load(best_model_state_path)
    best_model = model
    best_model.load_state_dict(best_model_state)

    if test_data and evaluate_on_test:
        logger.info("The model will be evaluated using the best epoch weights.")
        test_metrics = evaluate(
                best_model, test_data, validation_iterator or iterator,
                cuda_device=trainer._cuda_devices[0] # pylint: disable=protected-access
        )
        for key, value in test_metrics.items():
            metrics["test_" + key] = value

    elif test_data:
        logger.info("To evaluate on the test set after training, pass the "
                    "'evaluate_on_test' flag, or use the 'allennlp evaluate' command.")

    dump_metrics(os.path.join(serialization_dir, "metrics.json"), metrics, log=True)

    return best_model

