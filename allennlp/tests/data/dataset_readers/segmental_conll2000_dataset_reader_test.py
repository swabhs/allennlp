# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers.segmental_conll2000 import SegmentalConll2000DatasetReader


class TestSegmentalConll2000DatasetReader():
    """
    Tests for SegmentalConll2000DatasetReader.
    """
    @pytest.mark.parametrize("lazy", (True, False))
    @pytest.mark.parametrize("use_segmental_labels", (True, False))
    @pytest.mark.parametrize("use_binary_labels", (True, False))
    def test_read_from_file(self, lazy, use_segmental_labels, use_binary_labels):
        """
        When maximum span width is very long.
        """
        reader = SegmentalConll2000DatasetReader(lazy=lazy,
                                                 use_binary_labels=use_binary_labels,
                                                 use_segmental_labels=use_segmental_labels)
        instances = reader.read(str(AllenNlpTestCase.FIXTURES_ROOT / 'data' / 'chunks_bioul.conll'))
        instances = ensure_list(instances)
        assert len(instances) == 3

        fields = instances[1].fields
        # Token         Tag     Ind L-Start L-End L-Map R-End
        # <S>           U-O     0   0       0     0     0
        # The           B-NP    1   1       1     1     5
        # Food          I-NP    2   1       2     1     5
        # and           I-NP    3   1       3     1     5
        # Drug          I-NP    4   1       4     1     5
        # AdministrationL-NP    5   1       5     1     5
        # had           B-VP    6   6       6     2     7
        # raised        L-VP    7   6       7     2     7
        # questions     U-NP    8   8       8     3     8
        # about         U-PP    9   9       9     4     9
        # the           B-NP    10  10      10    5     11
        # device        L-NP    11  10      11    5     11
        # 's            B-NP    12  12      12    6     13
        # design        L-NP    13  12      13    6     13
        # .             U-O     14  14      14    7     14
        # </S>          U-O     15  15      15    8     15

        # Tokens
        expected_tokens = ["<S>", "The", "Food", "and", "Drug",
                           "Administration", "had", "raised", "questions",
                           "about", "the", "device", "'s", "design", ".", "</S>"]
        tokens = [t.text for t in fields['tokens'].tokens]
        assert tokens == expected_tokens

        # BIOUL tags
        if not use_binary_labels and not use_segmental_labels:
            expected_labels = ["U-O",
                               "B-NP", "I-NP", "I-NP", "I-NP", "L-NP",
                               "B-VP", "L-VP",
                               "U-NP",
                               "U-PP",
                               "B-NP", "L-NP",
                               "B-NP", "L-NP",
                               "U-O",
                               "U-O"]
        elif use_segmental_labels and not use_binary_labels:
            expected_labels = ["O",
                               "NP", "NP", "NP", "NP", "NP",
                               "VP", "VP",
                               "NP",
                               "PP",
                               "NP", "NP",
                               "NP", "NP",
                               "O",
                               "O"]
        elif use_binary_labels:
            expected_labels = ["U",
                               "B", "I", "I", "I", "L",
                               "B", "L",
                               "U",
                               "U",
                               "B", "L",
                               "B", "L",
                               "U",
                               "U"]
        assert fields["tags"].labels == expected_labels

        # Segment boundaries and mapping
        expected_seg_begs = [0, 1, 6, 8, 9, 10, 12, 14, 15]
        expected_seg_ends = [0, 5, 7, 8, 9, 11, 13, 14, 15]
        expected_seg_start_map = [0, 1, 1, 1, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7, 8]

        assert [l.sequence_index for l in fields['seg_starts'].field_list] == expected_seg_begs
        assert [l.sequence_index for l in fields['seg_ends'].field_list] == expected_seg_ends
        assert [l.sequence_index for l in fields['seg_map'].field_list] == expected_seg_start_map
