# pylint: disable=no-self-use,invalid-name
from collections import defaultdict

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.chunky.chunky_elmo_indexer import ChunkyElmoIndexer


class TestChunkyElmoIndexer(AllenNlpTestCase):
    # def setUp(self):
    #     super(TestChunkyElmoIndexer, self).setUp()

    def test_token_to_indices(self):

        indexer = ChunkyElmoIndexer()
        sentence = "I have a good dog ."
        indices = indexer.tokens_to_indices([Token(word) for word in sentence.split()])

        assert False



