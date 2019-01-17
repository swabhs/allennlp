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
    def __init__(self, segmental_path: str):
        super(ChunkyElmoTokenEmbedder, self).__init__()

        self.seglm = load_archive(segmental_path).model
        # self.output_dim = self.seglm.get_output_dim()

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        """
        output_dict = self.chunker({"elmo": inputs})
        chunk_tags = output_dict["tags"]
        # TODO: map input into right kind of tensor for seg-lm.

        # TODO: detach tensors??? - Matt
        lm_output_dict = self.seglm(inputs, chunk_tags)

        return lm_output_dict
        # raise NotImplementedError

    def get_output_dim(self) -> int:
        """
        """
        # raise NotImplementedError
        return self.output_dim

