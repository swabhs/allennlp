# pylint: disable=no-self-use,invalid-name
from collections import defaultdict

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.data.dataset import Batch, Instance
from allennlp.data.iterators.basic_iterator import BasicIterator
from allennlp.data.fields.text_field import TextField
from allennlp.chunky.chunky_elmo_indexer import ChunkyElmoIndexer

from calypso.labeled_seglm_transformer import LabeledSegLMTransformer


class TestChunkyElmoIndexer(AllenNlpTestCase):
    def setUp(self):
        super(TestChunkyElmoIndexer, self).setUp()
        self.indexer = ChunkyElmoIndexer(
            chunker_path="/Users/swabhas/pretrained/log_chunking_ptb_comparable/model.tar.gz",
            segmental_path="/Users/swabhas/pretrained/log_1b_labeled_seglm_transformer/model.tar.gz")

    def test_token_to_indices(self):
        sentence = "Recent work examines the extent to which RNN-based models capture" \
                    " syntax-sensitive phenomena that are traditionally taken as evidence for" \
                    " the existence in hierarchical structure ."
        tokens = [Token(word) for word in sentence.split()]
        indices = self.indexer.tokens_to_indices(tokens, Vocabulary(), "test-chunky")

        assert set(indices.keys()) == {"mask", "character_ids", "tags", "seg_map", "seg_starts","seg_ends"}


        tag_str = [self.indexer.seglm_vocab.get_token_from_index(i, "labels") for i in indices["tags"]]

        # Check for valid BIO labels.
        for i, tag in enumerate(tag_str):
            if i == 0:
                assert tag.startswith('B') or tag.startswith('U')
            if tag.startswith('I'):
                label = tag[1:]
                assert tag_str[i-1] in ['I'+label, 'B'+label]
            if tag.startswith('L'):
                label = tag[1:]
                assert tag_str[i-1] in ['I'+label, 'B'+label]
            if i > 0 and (tag.startswith('O') or tag.startswith('B') or tag.startswith('U')):
                assert tag_str[i-1].startswith('L') or tag_str[i-1].startswith('U') or tag_str[i-1].startswith('O')

        # TODO: How to ensure actual input tensor with the right vocabulary and indices? Matt: Don't worry about it.

    def test_token_to_indices_batch_size_2(self):
        # Checks whether or not AllenNLP overwrites padding logic.

        # Test with batch size 2 with different lengths.
        batch_sentences = ["I have a good dog called Killi .", "He fetches me stuff ."]

        instances = []
        for sentence in batch_sentences:
            tokens = [Token(token) for token in sentence.split()]
            field = TextField(tokens, {'test_chunky': self.indexer})
            instance = Instance({"elmo_chunky": field})
            instances.append(instance)

        vocab = Vocabulary()
        iterator = BasicIterator()
        iterator.index_with(vocab)

        for batch in iterator(instances, num_epochs=1, shuffle=False):
            break

        assert (batch['elmo_chunky']['mask'] > 0).sum(dim=1).tolist() == [8, 5]
        assert (batch['elmo_chunky']['seg_map'] > -1).sum(dim=1).tolist() == [8, 5]
        assert ((batch['elmo_chunky']['character_ids'] > 0).sum(dim=2) == 50).sum(dim=1).tolist() == [8, 5]