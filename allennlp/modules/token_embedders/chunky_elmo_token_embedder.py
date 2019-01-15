import logging
from typing import Dict, List

import torch

from allennlp.common import Registrable
from allennlp.common.util import prepare_environment
from allennlp.models.archival import load_archive
from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.modules.segmental_elmo import SegmentalElmo


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@TokenEmbedder.register("chunky_elmo_token_embedder")
class ChunkyElmoTokenEmbedder(TokenEmbedder):
    """
    """
    def __init__(self,
                 chunker_path: str,
                 segmental_lm_weights: str,
                 segmental_lm_options: str):
        super(ChunkyElmoTokenEmbedder, self).__init__()
        logger.info("Reading Chunker from %s", chunker_path)
        chunker_archive = load_archive(chunker_path)
        prepare_environment(chunker_archive.config)
        self.chunker = chunker_archive.model
        for param in self.chunker.parameters():
            param.requires_grad_(False)
        self.chunker.eval()

        self.seglm = SegmentalElmo(segmental_lm_options, segmental_lm_weights, 1)
        self.output_dim = self.seglm.get_output_dim()

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

    def get_tensors_from_chunk_tags(self,
                                    chunk_tags: List[List[int]]):
        chunk_tags_str = []
        for tag_seq in chunk_tags:
            tag_seq_str = [self.chunker.vocab._index_to_token["labels"][tag] for tag in tag_seq]
            tag_seq_str_replaced = ['U-O' if tag == 'O' else tag for tag in tag_seq_str]
            chunk_tags_str.append(tag_seq_str_replaced)

        tensor_list = []
        for chunk_tags in chunk_tags_str:
            instance_fields = {}
            seg_starts = []
            seg_ends = []
            seg_map = []
            seg_count = 0
            for i, tag in enumerate(chunk_tags):
                if tag.startswith('B-') or tag.startswith('U-'):
                    start = i
                    seg_starts.append(start)
                if tag.startswith('L-') or tag.startswith('U-'):
                    end = i
                    assert end - start < self.seglm._max_span_width
                    seg_ends.append(end)
                    seg_map += [seg_count for _ in range(start, end+1)]
                    seg_count += 1

            instance_fields['seg_ends'] = seg_ends
            instance_fields['seg_starts'] = seg_starts
            instance_fields['seg_map'] = seg_map

            tensor_list.append(instance_fields)
        # TODO: convert tags, seg_starts, seg_ends, seg_map into tensors
        return tensor_list