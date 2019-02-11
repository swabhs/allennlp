import logging
from typing import Dict, List

import torch

from allennlp.common.util import prepare_environment
from allennlp.models.archival import load_archive
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.nn.util import remove_sentence_boundaries


@TokenEmbedder.register("chunky_elmo_token_embedder")
class ChunkyElmoTokenEmbedder(TokenEmbedder):
    """
    """
    def __init__(self,
                 segmental_path: str,
                 dropout: float = 0.0,
                 keep_lm_dropout: bool = True,
                 use_scalar_mix: bool = True,
                 requires_grad: bool = False):
        super().__init__()
        self.seglm = load_archive(segmental_path).model

        # Delete the SegLM softmax parameters -- not required, and helps save memory.
        # TODO(Swabha): Is this really doing what I want it to do?
        del self.seglm.softmax.softmax_W
        del self.seglm.softmax.softmax_b

        # Updating SegLM parameters, optionally.
        for param in self.seglm.parameters():
            param.requires_grad_(requires_grad)

        # TODO(Swabha): Zeroing out dropout is not something being done in other embedders, so I will skip.
        if not keep_lm_dropout:
            self.zero_out_seglm_dropout()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

        # TODO(Swabha): Actually set this to number of layers in SegLM Transformer
        # For now, we mix all layers from the base encoders and the top layer from the segmental encoders.
        num_layers = self.seglm._encoder.num_layers + 2

        self.use_scalar_mix = use_scalar_mix
        if use_scalar_mix:
            self._scalar_mix = ScalarMix(mixture_size=num_layers,
                                         do_layer_norm=False,
                                         trainable=True)

        # TODO(Swabha): Ask Brendan about some hack in the LanguageModelTokenEmbedder.

    def forward(self,  # pylint: disable=arguments-differ
                character_ids: torch.Tensor,
                mask: torch.Tensor,
                mask_with_bos_eos: torch.Tensor,
                seg_ends: torch.Tensor,
                seg_map: torch.Tensor,
                seg_starts: torch.Tensor,
                tags: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        """
        # TODO(Swabha/Matt): detach tensors??? - Matt
        args_dict = {"character_ids": character_ids,
                     "mask": mask_with_bos_eos,
                     "seg_ends": seg_ends,
                     "seg_map": seg_map,
                     "seg_starts": seg_starts,
                     "tags": tags}
        lm_output_dict = self.seglm(**args_dict)

        sequential_embeddings = lm_output_dict["sequential"]
        segmental_embeddings = lm_output_dict["segmental"]
        projection_embeddings = lm_output_dict["projection"]
        activations = [x.squeeze(1) for x in lm_output_dict["activations"]]

        embeddings_list = [segmental_embeddings, projection_embeddings] + activations
        if self.use_scalar_mix:
            averaged_embeddings = self._dropout(self._scalar_mix(embeddings_list))
        else:
            averaged_embeddings = projection_embeddings

        averaged_embeddings_no_bos_eos, _ = remove_sentence_boundaries(averaged_embeddings, mask_with_bos_eos)
        return averaged_embeddings_no_bos_eos

    def get_output_dim(self) -> int:
        return self.seglm.get_output_dim()

    def zero_out_seglm_dropout(self):
        " TODO(Swabha): Probably remove..."
        self.seglm._dropout.p = 0.0
        self.seglm._encoder._contextual_encoder._dropout.p = 0.0
        self.seglm._segmental_encoder_bwd._dropout.p = 0.0
        self.seglm._segmental_encoder_fwd._dropout.p = 0.0

        transformers = [self.seglm._segmental_encoder_bwd._backward_transformer,
                      self.seglm._segmental_encoder_fwd._forward_transformer,
                      self.seglm._encoder._contextual_encoder._backward_transformer,
                      self.seglm._encoder._contextual_encoder._forward_transformer]

        for transformer  in transformers:
            for layer in transformer.layers:
                layer.self_attn.dropout.p = 0.0
                layer.feed_forward.dropout.p = 0.0
                for sub in layer.sublayer:
                    sub.dropout.p = 0.0

