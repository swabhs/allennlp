# pylint: disable=no-self-use,invalid-name
from collections import defaultdict

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.chunky.chunky_elmo_indexer import ChunkyElmoIndexer


class TestChunkyElmoIndexer(AllenNlpTestCase):
    # def setUp(self):
    #     super(TestChunkyElmoIndexer, self).setUp()

    def test_token_to_indices(self):

        indexer = ChunkyElmoIndexer("/Users/swabhas/pretrained/log_chunking_ptb_comparable/model.tar.gz")
        sentence = "I have a good dog ."
        indices = indexer.tokens_to_indices([Token(word) for word in sentence.split()], Vocabulary(), "test-chunky")

        print(indices)

        # TODO: actual tensors with the right vocabulary and indices.

        # TODO: create test with batch size 2 with different lengths.

        assert False


    def test_token_to_indices_batch_size_2(self):

        sentences = ["I have a good dog .", "He fetches me stuff ."]


        instances = []
        indexer = ChunkyElmoIndexer("/Users/swabhas/pretrained/log_chunking_ptb_comparable/model.tar.gz")
        for sentence in batch:
            tokens = [Token(token) for token in sentence]
            field = TextField(tokens,
                            {'character_ids': indexer})
            instance = Instance({"elmo": field})
            instances.append(instance)

        dataset = Batch(instances)
        vocab = Vocabulary()
        dataset.index_instances(vocab)
        dataset.as_tensor_dict()['elmo']['character_ids']


        # TODO prolly overwrite padding logice, but hopeuflly not.
        assert False