# pylint: disable=invalid-name,arguments-differ,abstract-method
import numpy as np
import pytest
import torch

from allennlp.common.testing import ModelTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.models import Model
from allennlp.models.segmental_language_model import SegmentalLanguageModel


class TestSegmentalLanguageModel(ModelTestCase):
    def setUp(self):
        super().setUp()

        self.expected_embedding_shape = (3, 21, 46)
        self.set_up_model(self.FIXTURES_ROOT / 'segmental_language_model' / 'experiment.jsonnet',
                          self.FIXTURES_ROOT / 'data' / 'chunks_bioul.conll')

    # pylint: disable=no-member
    def test_segmental_language_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent(keys_to_ignore=["batch_weight"])

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        result = self.model(**training_tensors)

        assert set(result) == {"loss", "forward_loss", "backward_loss", "lm_embeddings",
                               "noncontextual_token_embeddings", "mask", "batch_weight",
                               "projection"}

        # The model should preserve the BOS / EOS tokens.
        embeddings = result["projection"]
        assert tuple(embeddings.shape) == self.expected_embedding_shape

        loss = result["loss"].item()
        forward_loss = result["forward_loss"].item()
        backward_loss = result["backward_loss"].item()
        np.testing.assert_almost_equal(loss, (forward_loss + backward_loss) / 2,
                                           decimal=3)


    def test_get_gathered_embeddings_shorter(self):
        embeddings = torch.Tensor([
            [[0.0842, 0.8478], [0.9810, 0.3842], [0.3279, 0.7760], [0.8819, 0.7125]],
            [[0.0822, 0.6478], [0.8478, 0.5672], [0.8910, 0.0000], [0.0000, 0.0000]],
            [[0.0312, 0.6372], [0.8983, 0.4015], [0.0000, 0.0000], [0.0000, 0.0000]]
            ])
        indices = torch.LongTensor([[[0], [2], [3]],
                                    [[0], [1], [2]],
                                    [[0], [1], [-1]]
                                    ])

        expected_gathered = torch.Tensor([
            [[0.0842, 0.8478], [0.3279, 0.7760], [0.8819, 0.7125]],
            [[0.0822, 0.6478], [0.8478, 0.5672], [0.8910, 0.0000]],
            [[0.0312, 0.6372], [0.8983, 0.4015], [0.0312, 0.6372]]
            ])
        expected_mask = torch.LongTensor([[1, 1, 1],
                                          [1, 1, 1],
                                          [1, 1, 0]
                                          ])

        gathered, mask = SegmentalLanguageModel._get_gathered_embeddings(embeddings, indices)

        self.assertTrue(bool((expected_gathered == gathered).all().item()))
        self.assertTrue(bool((expected_mask == mask).all().item()))

    def test_get_gathered_embeddings_longer(self):
        embeddings = torch.Tensor([
            [[0.0842, 0.8478], [0.9810, 0.3842], [0.3279, 0.7760], [0.8819, 0.7125]],
            [[0.0822, 0.6478], [0.8478, 0.5672], [0.8910, 0.0000], [0.0000, 0.0000]],
            [[0.0312, 0.6372], [0.8983, 0.4015], [0.0000, 0.0000], [0.0000, 0.0000]]
            ])
        indices = torch.LongTensor([[[0], [0], [1], [1], [1], [2], [2], [3]],
                                    [[0], [1], [2], [2], [2], [3], [-1], [-1]],
                                    [[0], [0], [0], [0], [1], [-1], [-1], [-1]]
                                    ])

        expected_gathered = torch.Tensor([
            [[0.0842, 0.8478], [0.0842, 0.8478], [0.9810, 0.3842], [0.9810, 0.3842], [0.9810, 0.3842], [0.3279, 0.7760], [0.3279, 0.7760], [0.8819, 0.7125]],
            [[0.0822, 0.6478], [0.8478, 0.5672], [0.8910, 0.0000], [0.8910, 0.0000], [0.8910, 0.0000], [0.0000, 0.0000], [0.0822, 0.6478], [0.0822, 0.6478]],
            [[0.0312, 0.6372], [0.0312, 0.6372], [0.0312, 0.6372], [0.0312, 0.6372], [0.8983, 0.4015], [0.0312, 0.6372], [0.0312, 0.6372], [0.0312, 0.6372],]
            ])

        gathered, _ = SegmentalLanguageModel._get_gathered_embeddings(embeddings, indices)
        self.assertTrue(bool((expected_gathered == gathered).all().item()))

        self.assertTrue(bool((expected_gathered == gathered).all().item()))