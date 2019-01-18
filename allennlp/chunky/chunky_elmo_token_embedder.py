import logging
from typing import Dict, List

import torch

from allennlp.common import Registrable
from allennlp.common.util import prepare_environment
from allennlp.models.archival import load_archive
from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.chunky.segmental_elmo import SegmentalElmo


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@TokenEmbedder.register("chunky_elmo_token_embedder")
class ChunkyElmoTokenEmbedder(TokenEmbedder):
    """
    """
    def __init__(self, segmental_path: str, projection_dim: int = 512):
        super(ChunkyElmoTokenEmbedder, self).__init__()

        self.seglm = load_archive(segmental_path).model
        self.output_dim = projection_dim

    def forward(self,  # pylint: disable=arguments-differ
                character_ids: torch.Tensor,
                mask: torch.Tensor,
                tags: torch.Tensor,
                seg_ends: torch.Tensor,
                seg_starts: torch.Tensor,
                seg_map: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        """
        # TODO: detach tensors??? - Matt
        lm_output_dict = self.seglm(character_ids, mask, tags, seg_ends, seg_starts, seg_map)

        return lm_output_dict["projection"]

    def get_output_dim(self) -> int:
        """
        """
        # raise NotImplementedError
        return self.output_dim

