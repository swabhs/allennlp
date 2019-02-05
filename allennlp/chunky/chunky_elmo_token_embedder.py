import logging
from typing import Dict, List

import torch

from allennlp.common import Registrable
from allennlp.common.util import prepare_environment
from allennlp.models.archival import load_archive
from allennlp.modules.token_embedders import TokenEmbedder


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@TokenEmbedder.register("chunky_elmo_token_embedder")
class ChunkyElmoTokenEmbedder(TokenEmbedder):
    """
    """
    def __init__(self,
                 segmental_path: str,
                 update_seglm_params: bool = False,
                 projection_dim: int = 1024):
        super(ChunkyElmoTokenEmbedder, self).__init__()
        self.seglm = load_archive(segmental_path).model

        # Delete the softmax parameters -- not required.
        del self.seglm.softmax.softmax_W
        del self.seglm.softmax.softmax_b

        if not update_seglm_params:
            # Backproping into these embeddings reduces performance...
            for param in self.seglm.parameters():
                param.requires_grad_(False)
            self.zero_out_seglm_dropout()

        self.output_dim = projection_dim

    def forward(self,  # pylint: disable=arguments-differ
                character_ids: torch.Tensor,
                mask: torch.Tensor,
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
                     "mask": mask,
                     "seg_ends": seg_ends,
                     "seg_map": seg_map,
                     "seg_starts": seg_starts,
                     "tags": tags}
        lm_output_dict = self.seglm(**args_dict)
        return lm_output_dict["projection"]

    def get_output_dim(self) -> int:
        """
        """
        return self.output_dim


    def zero_out_seglm_dropout(self):
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

