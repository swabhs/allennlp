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
    def __init__(self, chunker_path: str, segmental_lm_path: str):
        super(ChunkyElmoTokenEmbedder, self).__init__()

        logger.info("Reading Chunker")
        chunker_archive = load_archive(chunker_path)
        prepare_environment(chunker_archive.config)
        self.chunker = chunker_archive.model
        for param in self.chunker.parameters():
            param.requires_grad_(False)
        self.chunker.eval()

        self.seglm_path = segmental_lm_path
        self.output_dim = 1024

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        """
        # The chunker model is a CRF tagger (allennlp.models.crf_tagger)
        import ipdb; ipdb.set_trace()
        output_dict = self.chunker({"elmo": inputs})
        chunk_tags = output_dict["tags"]

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
