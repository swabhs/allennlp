# pylint: disable=no-self-use,invalid-name
from collections import defaultdict

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary


class TestChunkyElmoIndexer(AllenNlpTestCase):
    def setUp(self):
        super(TestChunkyElmoIndexer, self).setUp()
        