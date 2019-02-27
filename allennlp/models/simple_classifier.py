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
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 pre_encode_feedforward: FeedForward,
                 encoder: Seq2SeqEncoder,
                 integrator: Seq2SeqEncoder,
                 dropout: float,
                 integrator_dropout: float,
                 output_layer: Maxout, #Union[FeedForward, Maxout],
                 feedforward: Optional[FeedForward] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._dropout = torch.nn.Dropout(dropout)
        self._num_classes = self.vocab.get_vocab_size("labels")

        self._pre_encode_feedforward = pre_encode_feedforward
        self._encoder = encoder
        self._feedforward = feedforward
        self._integrator = integrator
        self._integrator_dropout = torch.nn.Dropout(integrator_dropout)

        if feedforward is not None:
            output_dim = feedforward.get_output_dim()
        else:
            output_dim = self._encoder.get_output_dim()

        self._combined_integrator_output_dim = self._integrator.get_output_dim()
        self._self_attentive_pooling_projection = Linear(self._combined_integrator_output_dim, 1)

        self._output_layer = output_layer
        check_dimensions_match(text_field_embedder.get_output_dim(),
                               self._pre_encode_feedforward.get_input_dim(),
                               "text field embedder output dim",
                               "Pre-encoder feedforward input dim")
        check_dimensions_match(self._pre_encode_feedforward.get_output_dim(),
                               self._encoder.get_input_dim(),
                               "Pre-encoder feedforward output dim",
                               "Encoder input dim")
        check_dimensions_match(self._encoder.get_output_dim() * 3,
                               self._integrator.get_input_dim(),
                               "Encoder output dim * 3",
                               "Integrator input dim")
        check_dimensions_match(self._integrator.get_output_dim() * 4,
                               self._output_layer.get_input_dim(),
                               "Integrator output dim * 4",
                               "Output layer input dim")
        check_dimensions_match(self._output_layer.get_output_dim(),
                               self._num_classes,
                               "Output layer output dim",
                               "Number of classes.")

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

        pre_encoded_text = self._pre_encode_feedforward(embedded_text_input)
        encoded_tokens = self._encoder(pre_encoded_text, mask)

        if self._dropout:
            encoded_tokens = self._dropout(encoded_tokens)

        if self._feedforward is not None:
            encoded_tokens = self._feedforward(encoded_tokens)

        attention_logits = encoded_tokens.bmm(encoded_tokens.permute(0, 2, 1).contiguous())
        attention_weights = util.masked_softmax(attention_logits, mask)
        encoded_text = util.weighted_sum(encoded_tokens, attention_weights)

        # Build the input to the integrator
        integrator_input = torch.cat([encoded_tokens,
                                      encoded_tokens - encoded_text,
                                      encoded_tokens * encoded_text], 2)
        integrated_encodings = self._integrator(integrator_input, mask)
        # Simple Pooling layers
        max_masked_integrated_encodings = util.replace_masked_values(
                integrated_encodings, mask.unsqueeze(2), -1e7)
        max_pool = torch.max(max_masked_integrated_encodings, 1)[0]
        min_masked_integrated_encodings = util.replace_masked_values(
                integrated_encodings, mask.unsqueeze(2), +1e7)
        min_pool = torch.min(min_masked_integrated_encodings, 1)[0]
        mean_pool = torch.sum(integrated_encodings, 1) / torch.sum(mask, 1, keepdim=True)

        # Self-attentive pooling layer
        # Run through linear projection. Shape: (batch_size, sequence length, 1)
        # Then remove the last dimension to get the proper attention shape (batch_size, sequence length).
        self_attentive_logits = self._self_attentive_pooling_projection(
                integrated_encodings).squeeze(2)
        self_weights = util.masked_softmax(self_attentive_logits, mask)
        self_attentive_pool = util.weighted_sum(integrated_encodings, self_weights)

        pooled_representations = torch.cat([max_pool, min_pool, mean_pool, self_attentive_pool], 1)
        pooled_representations_dropped = self._integrator_dropout(pooled_representations)

        logits = self._output_layer(pooled_representations_dropped)
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


