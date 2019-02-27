from typing import Dict, Optional, Union

import numpy
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear


from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Elmo, FeedForward, Maxout, Seq2SeqEncoder, TextFieldEmbedder, TimeDistributed
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("simple_classifier")
class SimpleClassifier(Model):
    """
    A simple classifier.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    embedding_dropout : ``float``
        The amount of dropout to apply on the embeddings.
    pre_encode_feedforward : ``FeedForward``
        A feedforward network that is run on the embedded tokens before they
        are passed to the encoder.
    encoder : ``Seq2SeqEncoder``
        The encoder to use on the tokens.
    integrator : ``Seq2SeqEncoder``
        The encoder to use when integrating the attentive text encoding
        with the token encodings.
    integrator_dropout : ``float``
        The amount of dropout to apply on integrator output.
    output_layer : ``Union[Maxout, FeedForward]``
        The maxout or feed forward network that takes the final representations and produces
        a classification prediction.
    elmo : ``Elmo``, optional (default=``None``)
        If provided, will be used to concatenate pretrained ELMo representations to
        either the integrator output (``use_integrator_output_elmo``) or the
        input (``use_input_elmo``).
    use_input_elmo : ``bool`` (default=``False``)
        If true, concatenate pretrained ELMo representations to the input vectors.
    use_integrator_output_elmo : ``bool`` (default=``False``)
        If true, concatenate pretrained ELMo representations to the integrator output.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 feedforward: Optional[FeedForward] = None,
                 dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._dropout = torch.nn.Dropout(dropout)
        self._num_classes = self.vocab.get_vocab_size("labels")

        self._encoder = encoder
        self._feedforward = feedforward

        if feedforward is not None:
            output_dim = feedforward.get_output_dim()
        else:
            output_dim = self._encoder.get_output_dim()
        self._output_layer = TimeDistributed(Linear(output_dim,
                                                    self._num_classes))

        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")

        self.metrics = {
                "accuracy": CategoricalAccuracy(),
                "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``.
        label : torch.LongTensor, optional (default = None)
            A variable representing the label for each instance in the batch.
        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a
            distribution over the label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        mask = util.get_text_field_mask(tokens).float()
        embedded_text_input = self._text_field_embedder(tokens)

        if self._dropout:
            embedded_text_input = self._dropout(embedded_text_input)

        encoded_text = self._encoder(embedded_text_input, mask)

        if self._dropout:
            encoded_text = self.dropout(encoded_text)

        if self._feedforward is not None:
            encoded_text = self._feedforward(encoded_text)

        logits = self._output_layer(encoded_text)

        class_probabilities = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {'logits': logits, 'class_probabilities': class_probabilities}
        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        predictions = output_dict["class_probabilities"].cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}


