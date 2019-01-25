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
                 projection_dim: int = 1024):
        super(ChunkyElmoTokenEmbedder, self).__init__()
        self.seglm = load_archive(segmental_path).model
        del self.seglm.softmax.softmax_W
        del self.seglm.softmax.softmax_b
        for param in self.seglm.parameters():
            param.requires_grad_(False)
        self.seglm.eval()
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

