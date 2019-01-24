import logging
from typing import Dict, List
import torch

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import pad_sequence_to_length, prepare_environment
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer, ELMoCharacterMapper
from allennlp.data.vocabulary import Vocabulary


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
                 segmental_path: str,
                 max_span_width: int = 89,
                 namespace: str = 'chunky_elmo') -> None:
        self._namespace = namespace
        self._max_span_width = max_span_width

        # First initialize the chunker.
        logger.info("Reading Chunker from %s", chunker_path)
        from allennlp.models.archival import load_archive
        chunker_archive = load_archive(chunker_path)
        self.chunker = chunker_archive.model
        for param in self.chunker.parameters():
            param.requires_grad_(False)
        self.chunker.eval()
        self.elmo_indexer = ELMoTokenCharactersIndexer(namespace='elmo_characters')

        self.seglm_vocab = load_archive(segmental_path).model.vocab

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        pass

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[List[int]]]:
        character_indices = self.elmo_indexer.tokens_to_indices(tokens, vocabulary, "elmo")
        # token_indices = self.get_token_ids_for_seglm(tokens)

        # TODO(Swabha): worry about cuda, cudifying the model and constructor
        character_indices_tensor = {"elmo": torch.LongTensor(character_indices["elmo"]).unsqueeze(0)}

        # The chunker model is a CRF tagger (allennlp.models.crf_tagger)
        output_dict = self.chunker(character_indices_tensor)
        chunk_tags = output_dict["tags"]

        # Get string chunk tags.
        chunk_tags_str, instance_fields = self.get_tensors_from_chunk_tags(chunk_tags[0])
        # Convert these into tags for the language model.
        chunk_tags_seglm_ids = self.get_tags_in_lm_vocab(chunk_tags_str)

        from collections import OrderedDict
        return_dict = OrderedDict(sorted({'character_ids': character_indices["elmo"],
                       'mask': [1] * len(tokens),
                       'tags': chunk_tags_seglm_ids}.update(instance_fields)))
        # return_dict

        return return_dict

    @overrides
    def get_padding_token(self) -> int:
        # TODO(Swabha): Exact replica of `openai_transformer_byte_pair_indexer`.
        return 0

    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:  # pylint: disable=unused-argument
        return {}

    @staticmethod
    def _default_value_for_character_id_padding():
        return [0] * ELMoCharacterMapper.max_word_length

    @staticmethod
    def _default_value_for_tags():
        return -1

    @staticmethod
    def _default_value_for_mask():
        return 0

    @overrides
    def pad_token_sequence(self,
                           tokens: Dict[str, List[int]],
                           desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]) -> Dict[str, List[int]]:  # pylint: disable=unused-argument
        ret = {}
        for key, default_val in [
            ['character_ids', self._default_value_for_character_id_padding],
            # ['token_ids', self._default_value_for_mask],
            ['seg_ends', self._default_value_for_tags],
            ['seg_starts', self._default_value_for_tags],
            ['seg_map', self._default_value_for_tags],
            ['tags', self._default_value_for_tags],
            ['mask', self._default_value_for_mask]
        ]:
            ret[key] = pad_sequence_to_length(tokens[key],
                                              desired_num_tokens[key],
                                              default_value=default_val)

        # ret['character_ids'] = pad_sequence_to_length(tokens['character_ids'], desired_num_tokens['character_ids'],
        #                                     default_value=self._default_value_for_character_id_padding)
        return ret

    def get_tensors_from_chunk_tags(self,
                                    chunk_tags: List[int]):

        chunk_tags_str = [self.chunker.vocab._index_to_token["labels"][tag] for tag in chunk_tags]
        # Logic from SegmentalConll2000DatasetReader
        chunk_tags_str = ['U-O' if tag == 'O' else tag for tag in chunk_tags_str]

        instance_fields = {}
        seg_starts = []
        seg_ends = []
        seg_map = []
        seg_count = 0
        for i, tag in enumerate(chunk_tags_str):
            if tag.startswith('B-') or tag.startswith('U-'):
                start = i
                seg_starts.append(start)
            if tag.startswith('L-') or tag.startswith('U-'):
                end = i
                assert end - start < self._max_span_width
                seg_ends.append(end)
                seg_map += [seg_count for _ in range(start, end+1)]
                seg_count += 1

        instance_fields['seg_ends'] = seg_ends
        instance_fields['seg_starts'] = seg_starts
        instance_fields['seg_map'] = seg_map

        return chunk_tags_str, instance_fields

    def get_tags_in_lm_vocab(self, chunk_tags_str:str):
        return [self.seglm_vocab.get_token_index(t, "labels") for t in chunk_tags_str]

    def get_token_ids_for_seglm(self, tokens: List[Token]):
        token_ids = [self.seglm_vocab.get_token_index(token.text, "lm") for token in tokens]
        return token_ids

if __name__== "__main__":
    from calypso.labeled_seglm_transformer import LabeledSegLMTransformer
    self = ChunkyElmoIndexer("/Users/swabhas/pretrained/log_chunking_ptb_comparable/model.tar.gz",
        "/Users/swabhas/pretrained/log_1b_labeled_seglm_transformer/model.tar.gz")
    sentence = "I have a good dog ."
    tokens =[Token(word) for word in sentence.split()]
    vocabulary = Vocabulary()
    index_name = "test-chunky"

    batch = ["I have a good dog .", "He fetches me stuff ."]

    from allennlp.data.dataset import Batch, Instance
    from allennlp.data.fields.text_field import TextField
    instances = []
    for sentence in batch:
        tokens = [Token(token) for token in sentence]
        field = TextField(tokens, {'test_chunky': self})
        instance = Instance({"elmo_chunky": field})
        instances.append(instance)

    dataset = Batch(instances)
    vocab = Vocabulary()
    dataset.index_instances(vocab)

