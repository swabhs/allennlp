import logging
from typing import Dict, List

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import pad_sequence_to_length, prepare_environment
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.vocabulary import Vocabulary
# from allennlp.models.archival import load_archive

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@TokenIndexer.register("chunky_elmo")
class ChunkyElmoIndexer(TokenIndexer[List[int]]):
    """
    Convert a token to an array of character ids to compute ELMo representations.

    Parameters
    ----------
    namespace : ``str``, optional (default=``elmo_characters``)
    """
    # pylint: disable=no-self-use
    def __init__(self,
                 chunker_path: str,
                 namespace: str = 'chunky_elmo') -> None:
        self._namespace = namespace

        # First initialize the chunker.
        logger.info("Reading Chunker from {}", chunker_path)
        # chunker_archive = load_archive(chunker_path)
        # prepare_environment(chunker_archive.config)
        # self.chunker = chunker_archive.model
        # for param in self.chunker.parameters():
        #     param.requires_grad_(False)
        # self.chunker.eval()
        self.elmo_indexer = ELMoTokenCharactersIndexer(namespace='elmo_characters')

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        pass

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[List[int]]]:
        # TODO(swabha): index with Elmo Indexer
        character_indices = self.elmo_indexer.tokens_to_indices(tokens, vocabulary)

        # call chunker
        # The chunker model is a CRF tagger (allennlp.models.crf_tagger)
        output_dict = self.chunker({"elmo": character_indices})
        chunk_tags = output_dict["tags"]

        # TODO(Swabha): get string chunk tags. convert these into tags for the language model.

    @overrides
    def get_padding_lengths(self, token: List[int]) -> Dict[str, int]:
        # pylint: disable=unused-argument
        return {}

    @overrides
    def get_padding_token(self) -> List[int]:
        return []

    @staticmethod
    def _default_value_for_padding():
        # TODO(swabha): rewrite
        return [0] * ELMoCharacterMapper.max_word_length

    @overrides
    def pad_token_sequence(self,
                           tokens: Dict[str, List[List[int]]],
                           desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]) -> Dict[str, List[List[int]]]:
        # pylint: disable=unused-argument
        # TODO(swabha): rewrite
        return {key: pad_sequence_to_length(val, desired_num_tokens[key],
                                            default_value=self._default_value_for_padding)
                for key, val in tokens.items()}

    def convert_chunk_tags_into_other_fields(chunk_tags):
        # TODO(swabha): will index fields etc. work here?
        seg_starts = []
        seg_ends = []
        seg_map = []

        seg_count = 0
        for i, tag in enumerate(chunk_tags):
            if tag.startswith('B-') or tag.startswith('U-'):
                start = i
                seg_starts.append(IndexField(start, sentence))
            if tag.startswith('L-') or tag.startswith('U-'):
                end = i
                assert end - start < self._max_span_width
                seg_ends.append(IndexField(end, sentence))
                seg_map += [
                    IndexField(seg_count, instance_fields[field_name]) for _ in range(start, end+1)]
                seg_count += 1

        return seg_starts, seg_ends, seg_map
