"""
A :class:`~allennlp.training.trainer.Trainer` is responsible for training a
:class:`~allennlp.models.model.Model`.

Typically you might create a configuration file specifying the model and
training parameters and then use :mod:`~allennlp.commands.train`
rather than instantiating a ``Trainer`` yourself.
"""

import logging
import os
import shutil
import time
from typing import Dict, Optional, List, Tuple, Union, Iterable

import torch
import torch.optim.lr_scheduler
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.optim.lr_scheduler import _LRScheduler as PytorchLRScheduler  # pylint: disable=protected-access
from tensorboardX import SummaryWriter

from allennlp.common import Params
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import peak_memory_mb, gpu_memory_mb, dump_metrics
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.archival import archive_model
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer import Trainer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ScaffoldedTrainer(Trainer):
    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 iterator: DataIterator,
                 train_dataset: Iterable[Instance],
                 aux_iterator: DataIterator,
                 aux_train_dataset: Iterable[Instance],
                 mixing_ratio: float = 1.0,
                 cutoff_epoch: int = -1,
                 validation_dataset: Optional[Iterable[Instance]] = None,
                 patience: Optional[int] = None,
                 validation_metric: str = "-loss",
                 validation_iterator: DataIterator = None,
                 shuffle: bool = True,
                 num_epochs: int = 20,
                 serialization_dir: Optional[str] = None,
                 num_serialized_models_to_keep: int = 20,
                 keep_serialized_model_every_num_seconds: int = None,
                 model_save_interval: float = None,
                 cuda_device: Union[int, List] = -1,
                 grad_norm: Optional[float] = None,
                 grad_clipping: Optional[float] = None,
                 learning_rate_scheduler: Optional[LearningRateScheduler] = None,
                 summary_interval: int = 100,
                 histogram_interval: int = None,
                 should_log_parameter_statistics: bool = True,
                 should_log_learning_rate: bool = False) -> None:
        """
        Parameters
        ----------
        model : ``Model``, required.
            An AllenNLP model to be optimized. Pytorch Modules can also be optimized if
            their ``forward`` method returns a dictionary with a "loss" key, containing a
            scalar tensor representing the loss function to be optimized.
        optimizer : ``torch.nn.Optimizer``, required.
            An instance of a Pytorch Optimizer, instantiated with the parameters of the
            model to be optimized.
        iterator : ``DataIterator``, required.
            A method for iterating over a ``Dataset``, yielding padded indexed batches.
        train_dataset : ``Dataset``, required.
            A ``Dataset`` to train on. The dataset should have already been indexed.
        validation_dataset : ``Dataset``, optional, (default = None).
            A ``Dataset`` to evaluate on. The dataset should have already been indexed.
        patience : int, optional (default=2)
            Number of epochs to be patient before early stopping.
        validation_metric : str, optional (default="loss")
            Validation metric to measure for whether to stop training using patience
            and whether to serialize an ``is_best`` model each epoch. The metric name
            must be prepended with either "+" or "-", which specifies whether the metric
            is an increasing or decreasing function.
        num_epochs : int, optional (default = 20)
            Number of training epochs.
        serialization_dir : str, optional (default=None)
            Path to directory for saving and loading model files. Models will not be saved if
            this parameter is not passed.
        cuda_device : int, optional (default = -1)
            An integer specifying the CUDA device to use. If -1, the CPU is used.
            Multi-gpu training is not currently supported, but will be once the
            Pytorch DataParallel API stabilises.
        grad_norm : float, optional, (default = None).
            If provided, gradient norms will be rescaled to have a maximum of this value.
        grad_clipping : ``float``, optional (default = ``None``).
            If provided, gradients will be clipped `during the backward pass` to have an (absolute)
            maximum of this value.  If you are getting ``NaNs`` in your gradients during training
            that are not solved by using ``grad_norm``, you may need this.
        learning_rate_scheduler : PytorchLRScheduler, optional, (default = None)
            A Pytorch learning rate scheduler. The learning rate will be decayed with respect to
            this schedule at the end of each epoch. If you use
            :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`, this will use the ``validation_metric``
            provided to determine if learning has plateaued.
        no_tqdm : ``bool``, optional (default=False)
            We use ``tqdm`` for logging, which will print a nice progress bar that updates in place
            after every batch.  This is nice if you're running training on a local shell, but can
            cause problems with log files from, e.g., a docker image running on kubernetes.  If
            ``no_tqdm`` is ``True``, we will not use tqdm, and instead log batch statistics using
            ``logger.info``, outputting a line at most every 10 seconds.
        """
        super(ScaffoldedTrainer, self).__init__(
            model=model,
            optimizer=optimizer,
            iterator=iterator,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            patience=patience,
            validation_metric=validation_metric,
            validation_iterator=validation_iterator,
            shuffle=shuffle,
            num_epochs=num_epochs,
            serialization_dir=serialization_dir,
            cuda_device=cuda_device,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping,
            num_serialized_models_to_keep=num_serialized_models_to_keep,
            keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds,
            model_save_interval=model_save_interval,
            summary_interval=summary_interval,
            histogram_interval=histogram_interval,
            should_log_parameter_statistics=should_log_parameter_statistics,
            should_log_learning_rate=should_log_learning_rate)

        self._aux_iterator = aux_iterator
        self._aux_train_dataset = aux_train_dataset
        self._mixing_ratio = mixing_ratio
        self._cutoff_epoch = cutoff_epoch

    @overrides
    def _batch_loss(self,
                    batch: torch.Tensor,
                    for_training: bool,
                    aux_batch: torch.Tensor = None) -> torch.Tensor:
        """
        Does a forward pass on the given batch and returns the ``loss`` value in the result.
        If ``for_training`` is `True` also applies regularization penalty.
        If an auxiliary batch is provided, also does a forward pass on it to get the loss,
        and offsets the two using a mixing ratio.
        """
        if self._multiple_gpu:
            output_dict = self._data_parallel(batch)
        else:
            batch = util.move_to_device(batch, self._cuda_devices[0])
            output_dict = self._model(**batch)

        try:
            loss = output_dict["loss"]
            if for_training:
                loss += self._model.get_regularization_penalty()
        except KeyError:
            if for_training:
                raise RuntimeError("The model you are trying to optimize does not contain a"
                                   " 'loss' key in the output of model.forward(inputs).")
            loss = None

        if aux_batch is not None:
            aux_output_dict = self._model(**aux_batch)
            try:
                aux_loss = aux_output_dict["loss"]
                if for_training:
                    aux_loss += self._model.get_regularization_penalty()
            except KeyError:
                if for_training:
                    raise RuntimeError("The AUXILIARY model you are trying to optimize does not contain a"
                                        " 'loss' key in the output of model.forward(inputs).")
            loss += self._mixing_ratio * aux_loss

        return loss

    def _train_epoch(self, epoch: int) -> dict:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        logger.info(f"Peak CPU memory usage MB: {peak_memory_mb()}")
        for gpu, memory in gpu_memory_mb().items():
            logger.info(f"GPU {gpu} memory usage MB: {memory}")

        train_loss = 0.0
        # Set the model to "train" mode.
        self._model.train()

        # Get tqdm for the training batches
        train_generator = self._iterator(self._train_data,
                                         num_epochs=1,
                                         shuffle=self._shuffle,
                                         cuda_device=self._cuda_devices[0])
        num_training_batches = self._iterator.get_num_batches(self._train_data)
        self._last_log = time.time()
        last_save_time = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        if self._histogram_interval is not None:
            histogram_parameters = set(self._model.get_parameters_for_histogram_tensorboard_logging())

        logger.info("Training")
        train_generator_tqdm = Tqdm.tqdm(train_generator,
                                         total=num_training_batches)
        aux_train_generator = self._aux_iterator(self._aux_train_dataset,
                                                 num_epochs=1,
                                                 cuda_device=self._cuda_devices[0])

        for batch, aux_batch in zip(train_generator_tqdm, aux_train_generator):
            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            self._log_histograms_this_batch = self._histogram_interval is not None and (
                    batch_num_total % self._histogram_interval == 0)

            self._optimizer.zero_grad()

            loss = self._batch_loss(batch, for_training=True, aux_batch=aux_batch)
            loss.backward()

            train_loss += loss.item()

            batch_grad_norm = self._rescale_gradients()

            # This does nothing if batch_num_total is None or you are using an
            # LRScheduler which doesn't update per batch.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step_batch(batch_num_total)

            if self._log_histograms_this_batch:
                # get the magnitude of parameter updates for logging
                # We need a copy of current parameters to compute magnitude of updates,
                # and copy them to CPU so large models won't go OOM on the GPU.
                param_updates = {name: param.detach().cpu().clone()
                                 for name, param in self._model.named_parameters()}
                self._optimizer.step()
                for name, param in self._model.named_parameters():
                    param_updates[name].sub_(param.detach().cpu())
                    update_norm = torch.norm(param_updates[name].view(-1, ))
                    param_norm = torch.norm(param.view(-1, )).cpu()
                    self._tensorboard.add_train_scalar("gradient_update/" + name,
                                                       update_norm / (param_norm + 1e-7),
                                                       batch_num_total)
            else:
                self._optimizer.step()

            # Update the description with the latest metrics
            metrics = self._get_metrics(train_loss, batches_this_epoch)
            description = self._description_from_metrics(metrics)

            train_generator_tqdm.set_description(description, refresh=False)

            # Log parameter values to Tensorboard
            if batch_num_total % self._summary_interval == 0:
                if self._should_log_parameter_statistics:
                    self._parameter_and_gradient_statistics_to_tensorboard(batch_num_total, batch_grad_norm)
                if self._should_log_learning_rate:
                    self._learning_rates_to_tensorboard(batch_num_total)
                self._tensorboard.add_train_scalar("loss/loss_train", metrics["loss"], batch_num_total)
                self._metrics_to_tensorboard(batch_num_total,
                                             {"epoch_metrics/" + k: v for k, v in metrics.items()})

            if self._log_histograms_this_batch:
                self._histograms_to_tensorboard(batch_num_total, histogram_parameters)

            # Save model if needed.
            if self._model_save_interval is not None and (
                    time.time() - last_save_time > self._model_save_interval
            ):
                last_save_time = time.time()
                self._save_checkpoint(
                        '{0}.{1}'.format(epoch, time_to_str(int(last_save_time))), [], is_best=False
                )

        return self._get_metrics(train_loss, batches_this_epoch, reset=True)


    @classmethod
    def from_params(cls,
                    model: Model,
                    serialization_dir: str,
                    iterator: DataIterator,
                    aux_iterator: DataIterator,
                    train_dataset: Iterable[Instance],
                    aux_train_dataset: Iterable[Instance],
                    validation_dataset: Optional[Iterable[Instance]],
                    params: Params,
                    files_to_archive: Dict[str, str]) -> 'ScaffoldedTrainer':
        # pylint: disable=arguments-differ
        patience = params.pop_int("patience", None)
        validation_metric = params.pop("validation_metric", "-loss")
        shuffle = params.pop_bool("shuffle", True)
        num_epochs = params.pop_int("num_epochs", 20)
        cuda_device = params.pop_int("cuda_device", -1)
        grad_norm = params.pop_float("grad_norm", None)
        grad_clipping = params.pop_float("grad_clipping", None)
        lr_scheduler_params = params.pop("learning_rate_scheduler", None)

        if cuda_device >= 0:
            model = model.cuda(cuda_device)
        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))

        if lr_scheduler_params:
            scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params)
        else:
            scheduler = None

        num_serialized_models_to_keep = params.pop_int("num_serialized_models_to_keep", 20)
        keep_serialized_model_every_num_seconds = params.pop_int(
                "keep_serialized_model_every_num_seconds", None)
        model_save_interval = params.pop_float("model_save_interval", None)
        summary_interval = params.pop_int("summary_interval", 100)
        histogram_interval = params.pop_int("histogram_interval", None)
        should_log_parameter_statistics = params.pop_bool("should_log_parameter_statistics", True)
        should_log_learning_rate = params.pop_bool("should_log_learning_rate", False)

        # Scaffolding parameters.
        mixing_ratio = params.pop("mixing_ratio", 1.0)
        cutoff_epoch = params.pop("cutoff_epoch", -1)

        params.assert_empty(cls.__name__)
        return ScaffoldedTrainer(model=model,
                               optimizer=optimizer,
                               iterator=iterator,
                               train_dataset=train_dataset,
                               aux_iterator=aux_iterator,
                               aux_train_dataset=aux_train_dataset,
                               mixing_ratio=mixing_ratio,
                               cutoff_epoch=cutoff_epoch,
                               validation_dataset=validation_dataset,
                               patience=patience,
                                validation_metric=validation_metric,
                                validation_iterator=validation_iterator,
                                shuffle=shuffle,
                                num_epochs=num_epochs,
                                serialization_dir=serialization_dir,
                                cuda_device=cuda_device,
                                grad_norm=grad_norm,
                                grad_clipping=grad_clipping,
                                learning_rate_scheduler=scheduler,
                                num_serialized_models_to_keep=num_serialized_models_to_keep,
                                keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds,
                                model_save_interval=model_save_interval,
                                summary_interval=summary_interval,
                                histogram_interval=histogram_interval,
                                should_log_parameter_statistics=should_log_parameter_statistics,
                                should_log_learning_rate=should_log_learning_rate)
