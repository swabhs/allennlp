import numpy
import torch

from allennlp.training.metrics.metric import Metric
from allennlp.common.checks import ConfigurationError

@Metric.register("perplexity")
class Perplexity(Metric):
    """
    Computes perplexity for language modeling evaluation.
    """
    def __init__(self) -> None:
        self._total_loss = 0.0
        self._total_targets = 0.0

    def __call__(self,
                 loss: torch.Tensor,
                 num_targets: torch.Tensor):
        """
        loss: ``torch.Tensor``, required.
            Batch loss of shape (1).
        num_targets: ``torch.Tensor``, required.
            Number of targets in batch, shape (1).
        """
        if isinstance(loss, torch.Tensor):
            self._total_loss += loss.detach().item()
            self._total_targets += num_targets.detach().item()
        else:
            self._total_loss += loss
            self._total_targets += num_targets

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A dict with perplexity.
        """
        if self._total_targets <= 0:
            raise ConfigurationError("Number of targets in perplexity computation must be strictly "
                                     "positive, but was given {}.".format(self._total_targets))
        perplexity = numpy.exp(self._total_loss/self._total_targets)
        if reset:
            self.reset()
        return {'perplexity': perplexity}

    def reset(self):
        self._total_loss = 0.0
        self._total_targets = 0.0
