from typing import Dict, List, Tuple, Union

import torch
from torch.nn.modules.linear import Linear
import numpy as np

from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TimeDistributed
from allennlp.models.language_model import _SoftmaxLoss, LanguageModel
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.sampled_softmax_loss import SampledSoftmaxLoss
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics.perplexity import Perplexity


@Model.register('label_encoder_seglm')
class LabelEncoderSegLM(LanguageModel):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 contextualizer: Seq2SeqEncoder,
                 forward_segmental_contextualizer: Seq2SeqEncoder,
                 backward_segmental_contextualizer: Seq2SeqEncoder,
                 label_feature_dim: int,
                 softmax_projection_dim: int,
                 label_namespace: str = "labels",
                 dropout: float = None,
                 num_samples: int = None,
                 sparse_embeddings: bool = False,
                 bidirectional: bool = True,
                 initializer: InitializerApplicator = None) -> None:
        super().__init__(vocab=vocab,
                         text_field_embedder=text_field_embedder,
                         contextualizer=contextualizer,
                         dropout=dropout,
                         num_samples=num_samples,
                         sparse_embeddings=sparse_embeddings,
                         bidirectional=bidirectional,
                         initializer=initializer)
        self._forward_segmental_contextualizer = forward_segmental_contextualizer
        self._backward_segmental_contextualizer = backward_segmental_contextualizer

        if num_samples is not None:
            self._softmax_loss = SampledSoftmaxLoss(num_words=vocab.get_vocab_size(),
                                                    embedding_dim=softmax_projection_dim,
                                                    num_samples=num_samples,
                                                    sparse=sparse_embeddings)
        else:
            self._softmax_loss = _SoftmaxLoss(num_words=vocab.get_vocab_size(),
                                              embedding_dim=softmax_projection_dim)

        self.num_classes = self.vocab.get_vocab_size(label_namespace)
        self.label_feature_embedding = Embedding(self.num_classes, label_feature_dim)

        base_dim = contextualizer.get_output_dim() // 2
        seg_dim = base_dim + label_feature_dim
        self._forward_dim = softmax_projection_dim

        self.pre_segmental_layer = TimeDistributed(Linear(base_dim *2+label_feature_dim, softmax_projection_dim))
        self.projection_layer = TimeDistributed(Linear(base_dim *2, softmax_projection_dim))


    def num_layers(self) -> int:
        """
        Returns the depth of this LM. That is, how many layers the contextualizer has plus one for
        the non-contextual layer.
        """
        if hasattr(self._contextualizer, 'num_layers') and hasattr(self._backward_segmental_contextualizer, 'num_layers'):
            return self._contextualizer.num_layers + self._backward_segmental_contextualizer.num_layers + 1
        else:
            raise NotImplementedError(f"Contextualizer of type {type(self._contextualizer)} " +
                                      "does not report how many layers it has.")

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                tags: torch.Tensor,
                seg_map: torch.Tensor,
                seg_ends: torch.Tensor,
                seg_starts: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        Dict with keys:

        ``'loss'``: ``torch.Tensor``
            forward negative log likelihood, or the average of forward/backward
            if language model is bidirectional
        ``'forward_loss'``: ``torch.Tensor``
            forward direction negative log likelihood
        ``'backward_loss'``: ``torch.Tensor`` or ``None``
            backward direction negative log likelihood. If language model is not
            bidirectional, this is ``None``.
        ``'lm_embeddings'``: ``Union[torch.Tensor, List[torch.Tensor]]``
            (batch_size, timesteps, embed_dim) tensor of top layer contextual representations or
            list of all layers. No dropout applied.
        ``'noncontextual_token_embeddings'``: ``torch.Tensor``
            (batch_size, timesteps, token_embed_dim) tensor of bottom layer noncontextual
            representations
        ``'mask'``: ``torch.Tensor``
            (batch_size, timesteps) mask for the embeddings
        """
        # pylint: disable=arguments-differ
        mask = get_text_field_mask(tokens)

        # shape (batch_size, timesteps, embedding_size)
        contextual_embeddings = self._text_field_embedder(tokens)

        # # Either the top layer or all layers.
        # contextual_embeddings: Union[torch.Tensor, List[torch.Tensor]] = self._contextualizer(
        #         embeddings, mask
        # )

        return_dict = {'lm_embeddings': contextual_embeddings,
                    #    'noncontextual_token_embeddings': embeddings,
                       'mask': mask
                       }

        token_ids = tokens.get("tokens")
        if token_ids is None:
            return return_dict

        # If we have target tokens, calculate the loss.
        assert isinstance(contextual_embeddings, torch.Tensor)

        # Use token_ids to compute targets
        forward_targets = torch.zeros_like(token_ids)
        forward_targets[:, 0:-1] = token_ids[:, 1:]

        if self._bidirectional:
            backward_targets = torch.zeros_like(token_ids)
            backward_targets[:, 1:] = token_ids[:, 0:-1]
        else:
            backward_targets = None

        # add dropout
        sequential_forward, sequential_backward = contextual_embeddings.chunk(2, -1)

        # Lookup the label embeddings.
        embedded_label_indicator = self.label_feature_embedding(tags.long())
        # Label embeddings to be concatenated twice, so they feature once each
        # in the forward and backward losses.

        # seg_forward_input = self.pre_segmental_layer(torch.cat((sequential_forward, embedded_label_indicator), dim=-1))
        # Left -> Right direction:
        segmental_forward = self._get_segmental_embeddings(
            encoder=self._forward_segmental_contextualizer,
            unidirectional_embs=sequential_forward,
            embedded_label_indicator=embedded_label_indicator,
            boundaries=seg_starts,
            mapping=seg_map)
        # projected_seg_labeled_forward = self.seg_projection_layer(segmental_forward)
        projected_forward = self.projection_layer(torch.cat((sequential_forward,
                                                             segmental_forward), dim=-1))

        # seg_backward_input = self.pre_segmental_layer(torch.cat((sequential_backward, embedded_label_indicator), dim=-1))
        segmental_backward = self._get_segmental_embeddings(
            encoder=self._backward_segmental_contextualizer,
            unidirectional_embs=sequential_backward,
            embedded_label_indicator=embedded_label_indicator,
            boundaries=seg_ends,
            mapping=seg_map)
        # projected_seg_labeled_backward = self.seg_projection_layer(segmental_backward)
        projected_backward = self.projection_layer(torch.cat((sequential_backward,
                                                              segmental_backward), dim=-1))

        projected_bi = self._dropout(torch.cat((projected_forward,
                                                projected_backward), dim=-1))
        return_dict['projection'] = projected_bi
        # return_dict['segmental'] = torch.cat((projected_seg_labeled_forward,
        #                                       projected_seg_labeled_backward), dim=-1)

        # compute softmax loss
        # TODO(Swabha): What does embeddings do for loss computation?
        forward_loss, backward_loss = self._compute_loss(projected_bi,
                                                         contextual_embeddings,
                                                         forward_targets,
                                                         backward_targets)

        num_targets = torch.sum((forward_targets > 0).float())
        if num_targets > 0:
            if self._bidirectional:
                total_loss = 0.5 * (forward_loss + backward_loss)
                average_loss = total_loss / num_targets
            else:
                total_loss = forward_loss
                average_loss = forward_loss / num_targets
            return_dict.update({
                    'loss': average_loss,
                    'forward_loss': forward_loss / num_targets,
                    'backward_loss': (backward_loss / num_targets
                                        if backward_loss is not None else None),
                    'batch_weight': num_targets
            })
        else:
            # average_loss zero tensor, return it for all
            total_loss = average_loss = torch.tensor(0.0).to(forward_targets.device)  # pylint: disable=not-callable
            return_dict.update({
                    'loss': average_loss,
                    'forward_loss': average_loss,
                    'backward_loss': average_loss if backward_loss is not None else None
            })

        # this is stored to compute perplexity if needed
        self._last_average_loss[0] = average_loss.detach().item()
        # Send metrics for evaluation.
        self.metric(loss=total_loss, num_targets=num_targets)

        return return_dict

    def _get_segmental_embeddings(self,
                                  encoder: Seq2SeqEncoder,
                                  embedded_label_indicator: torch.tensor,
                                  unidirectional_embs: torch.tensor,
                                  boundaries: torch.LongTensor,
                                  mapping: torch.LongTensor):
        # First, collect the hidden states of segment boundaries.
        seg_boundary_embs, seg_boundary_mask = self._get_gathered_embeddings(
            embeddings=unidirectional_embs,
            indices=boundaries)

        seg_embeddings_scattered, seg_boundary_mask = self._get_gathered_embeddings(
            embeddings=seg_boundary_embs,
            indices=mapping)

        seg_encoder_input = self.pre_segmental_layer(torch.cat((seg_embeddings_scattered, unidirectional_embs, embedded_label_indicator), dim=-1))

        # Pass through forward or backward encoder.
        seg_embeddings = encoder(seg_encoder_input, seg_boundary_mask)

        # Secondly, the segmental embeddings need to be scattered, so each
        # position gets its own segmental information.

        return seg_embeddings

    @staticmethod
    def _get_gathered_embeddings(embeddings: torch.Tensor,
                                 indices: torch.LongTensor):
        """
        embeddings: shape (batch_size, seq_len, embedding_dim)
        indices: shape (batch_size, max_num_segments, 1)
        """
        batch_size, _, embedding_dim = embeddings.size()
        masked_indices = (indices.squeeze(-1) > -1).long()
        gatherable_indices = indices * masked_indices.view(batch_size, -1, 1)
        gatherable_indices = gatherable_indices.repeat(1, 1, embedding_dim)
        gathered_embeddings = embeddings.gather(dim=1, index=gatherable_indices)

        return gathered_embeddings, masked_indices

    def get_metrics(self, reset: bool = False):
        return self.metric.get_metric(reset=reset)
