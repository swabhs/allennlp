import logging
from typing import Dict

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
    def __init__(self, segmental_lm_path: str):
        super(ChunkyElmoTokenEmbedder, self).__init__()

        self.seglm_path = segmental_lm_path
        self.output_dim = 1024

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        """
        # TODO: detach tensors

        # TODO: convert `inputs` into `lm_targets`
        self.seglm = load_archive(self.seglm_path)
        seglm_config = self.seglm.config
        prepare_environment(seglm_config)
        seglm_model = self.seglm.model
        seglm_model.eval()
        lm_output_dict = seglm_model(inputs, chunk_tags)
        self.output_dim = lm_output_dict["emb"].size()[-1]



        return lm_output_dict
        # raise NotImplementedError

    def get_output_dim(self) -> int:
        """
        """
        # raise NotImplementedError
        return self.output_dim
