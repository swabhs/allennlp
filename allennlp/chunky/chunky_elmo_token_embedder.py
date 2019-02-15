import logging
from typing import Dict, List

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import prepare_environment
from allennlp.models.archival import load_archive
from allennlp.models.language_model import LanguageModel
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
                 concat_segmental: bool = False,
                 use_all_base_layers: bool = False,
                 use_projection_layer: bool = True,
                 use_scalar_mix: bool = True,
                 requires_grad: bool = False):
        super().__init__()
        self.seglm = load_archive(segmental_path).model

        # Delete the SegLM softmax parameters -- not required, and helps save memory.
        # TODO(Swabha): Is this really doing what I want it to do?
        if isinstance(self.seglm, LanguageModel):
            self.seglm.delete_softmax()
        else:
            del self.seglm.softmax.softmax_W
            del self.seglm.softmax.softmax_b

        # Updating SegLM parameters, optionally.
        for param in self.seglm.parameters():
            param.requires_grad_(requires_grad)

        # if remove_dropout:
        #     self.zero_out_seglm_dropout()
        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

        num_layers = 1  # for segmental-layers
        if use_all_base_layers:
            num_layers += self.seglm._contextualizer.num_layers
        else:
            num_layers += 1
        if use_projection_layer:
            num_layers += 1

        if use_scalar_mix:
            self._scalar_mix = ScalarMix(mixture_size=num_layers, do_layer_norm=False, trainable=True)
        else:
            self._scalar_mix = None

        # TODO(Swabha): Ask Brendan about some hack in the LanguageModelTokenEmbedder.
        self.use_all_base_layers = use_all_base_layers
        self.use_projection_layer = use_projection_layer
        self.concat_segmental = concat_segmental

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
        args_dict = {"mask": mask_with_bos_eos,
                     "seg_ends": seg_ends,
                     "seg_map": seg_map,
                     "seg_starts": seg_starts,
                     "tags": tags}
        if isinstance(self.seglm, LanguageModel):
            args_dict["tokens"] = {"elmo": character_ids}
        else:
            args_dict["character_ids"] = character_ids

        lm_output_dict = self.seglm(**args_dict)

        sequential_embeddings = lm_output_dict["sequential"]
        segmental_embeddings = lm_output_dict["segmental"]
        projection_embeddings = lm_output_dict["projection"]
        base_layer_embeddings = [emb.squeeze(1) for emb in lm_output_dict["activations"]]

        embeddings_list = []
        if self.use_all_base_layers:
            embeddings_list.append(base_layer_embeddings)
        else:
            embeddings_list.append(sequential_embeddings)

        # Always include segmental layer.
        embeddings_list.append(segmental_embeddings)

        if self.use_projection_layer:
            embeddings_list.append(projection_embeddings)

        if self._scalar_mix is None:
            averaged_embeddings = segmental_embeddings
        elif self.concat_segmental:
            averaged_embeddings = torch.cat((sequential_embeddings, segmental_embeddings), dim = -1)
        else:
            averaged_embeddings = self._dropout(self._scalar_mix(embeddings_list))

        averaged_embeddings_no_bos_eos, _ = remove_sentence_boundaries(averaged_embeddings, mask_with_bos_eos)
        return averaged_embeddings_no_bos_eos

    def get_output_dim(self) -> int:
        if isinstance(self.seglm, LanguageModel):
            return self.seglm._contextualizer.get_output_dim()
        return self.seglm.get_output_dim()

    def zero_out_seglm_dropout(self):
        # TODO(Swabha): Nothing like this in other token embedders, probably remove?
        if isinstance(self.seglm, LanguageModel):
            raise NotImplementedError
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

