from typing import Dict, List, Iterable
import itertools
import logging

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import to_bioul
from allennlp.data.fields import TextField, SequenceLabelField, Field, ListField, IndexField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _is_divider(line: str) -> bool:
    return line.strip() == ''


@DatasetReader.register('segmental_conll2000')
class SegmentalConll2000DatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    WORD POS-TAG CHUNK-TAG

    with a blank line indicating the end of each sentence
    and converts it into a ``Dataset`` suitable for segmental language modeling.

    An example looks like:
    Some DT B-NP
    analysts NNS L-NP
    add VBP U-VP
    that IN U-SBAR
    third-party JJ B-NP
    pressures NNS L-NP
    to TO B-VP
    reduce VB L-VP
    health NN B-NP
    costs NNS L-NP
    will MD B-VP
    continue VB I-VP
    to TO I-VP
    bedevil VB L-VP
    companies NNS U-NP
    ' POS B-NP
    bottom NN I-NP
    lines NNS L-NP
    . . O

    Each ``Instance`` contains the words in the ``"tokens"`` ``TextField``, as well as
    all possible spans in the sentence. The values corresponding to the ``tag_label`` for each
    token get loaded into the ``"tags"`` ``SequenceLabelField``. And if specified,
    any ``feature_labels`` (not recommended), the corresponding values will get loaded into
    their own ``SequenceLabelField`` s. This dataset reader ignores the "article" divisions and
    simply treats each sentence as an independent ``Instance``. (Technically the reader splits
    sentences on any combination of blank lines and "DOCSTART" tags; in particular,
    it does the right thing on well formed inputs.)

    This reader also splits the longer chunks into shorter chunks, of up to length
    ``"max_span_width"`` for runtime efficiency. All spans of text are loaded into the
    text are loaded into the ``"spans"`` field and their corresponding tags (without BIOUL encoding)
    are loaded into the ``"span-tags"`` field. We retain the spans with ``O`` tags, but limit them
    to be length-1 chunks.

    Parameters
    ----------
    lazy: ``bool``, optional (default=``False``)
        Read dataset lazily, reading more when required.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens":
                                                                       SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
     coding_scheme: ``str``, optional (default=``BIOUL``)
        Specifies the coding scheme for ``chunk_labels``. Valid options are ``BIO`` and ``BIOUL``.
        The ``BIOUL`` default requires modification of the CoNLL 2000 chunking data. In the BIO
        scheme, B is a tag starting a segment, I is a tag continuing a segment, and O is a tag
        outside of a segment. In the latter, U denotes a segment of length 1, and L is a tag end a
        segment.
    max_span_width: ``int``, optional (default=100)
        Maximum width of chunk segments to consider. Any chunk longer than this is split up.
    max_train_sentence_len: ``int``, optional (default=100)
        Maximum length of training sentence to consider, any sentence longer than this is skipped.
    label_namespace: ``str``, optional (default=``labels``)
        Specifies the namespace for the chosen ``tag_label``.
    use_segmental_labels: ``bool``, optional (default=`False`)
        Labels for segmental layers, hence don't need BIO scheme.
    use_binary_labels: ``bool``, optional (default=`False`)
        0-1 labels to indicate segment boundaries.
    """

    def __init__(self,
                 lazy: bool = True,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 coding_scheme: str = "BIOUL",
                 max_span_width: int = 100,
                 max_train_sentence_len: int = 100,
                 label_namespace: str = "labels",
                 use_segmental_labels: bool = False,
                 use_binary_labels: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

        if coding_scheme not in ("BIO", "BIOUL"):
            raise ConfigurationError("unknown coding_scheme: {}".format(coding_scheme))
        self.coding_scheme = coding_scheme
        self._original_coding_scheme = "BIOUL"

        self._max_span_width = max(max_span_width, max_train_sentence_len)
        self._max_train_sentence_len = max_train_sentence_len

        self.label_namespace = label_namespace
        self.use_segmental_labels = use_segmental_labels
        self.use_binary_labels = use_binary_labels

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as data_file:
            logger.info('Reading instances from lines in file at: %s', file_path)

            # Group into alternative divider / sentence chunks.
            for is_divider, lines in itertools.groupby(data_file, _is_divider):
                # Ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if is_divider:
                    continue
                fields = [line.strip().split() for line in lines]
                # unzipping trick returns tuples, but our Fields need lists
                tokens, _, chunk_tags = [list(field) for field in zip(*fields)]

                # HACK: Ignore long training sentences for memory efficiency.
                if "train" in file_path and len(tokens) > self._max_train_sentence_len:
                    continue
                # TextField requires ``Token`` objects
                # Since we are using these tokens in a language model we also need BOS/EOS tags.
                tokens = [Token('<S>')] + [Token(token) for token in tokens] + [Token('</S>')]
                chunk_tags = ['O'] + chunk_tags + ['O']

                yield self.text_to_instance(tokens, chunk_tags)

    def text_to_instance(self, # type: ignore
                         tokens: List[Token],
                         chunk_tags: List[str] = None) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        sentence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {'tokens': sentence}

        if chunk_tags is None:
            return Instance(instance_fields)
        chunk_tags = self.clip_chunks_by_max_length(chunk_tags)
        # Recode the labels if necessary.
        if self.coding_scheme == "BIOUL" and self._original_coding_scheme == "BIO":
            chunk_tags = to_bioul(chunk_tags, encoding=self._original_coding_scheme)

        # We want to treat O also as a valid span label, which is usually ignored.
        # However, each O span needs to be of length 1, since there is no reason to
        # combine tokens with O tags as a span, hence replacing O with U-O.
        chunk_tags = ['U-O' if tag == 'O' else tag for tag in chunk_tags]
        tags, field_name = self.convert_bioul_to_segmental(chunk_tags)
        instance_fields[field_name] = SequenceLabelField(tags, sentence, field_name)

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

        instance_fields['seg_ends'] = ListField(seg_ends)
        instance_fields['seg_starts'] = ListField(seg_starts)
        instance_fields['seg_map'] = ListField(seg_map)

        return Instance(instance_fields)

    def clip_chunks_by_max_length(self, chunk_sequence):
        """
        If the length of a chunk exceeds `self._max_span_width`, we split it into
        smaller chunks, each of length up to `self._max_span_width`. This is not detrimental
        if we select a long enough `self._max_span_width`, which would avoid clipping most spans.
        """
        clipped_chunks = [tag for tag in chunk_sequence]
        last_begin = 0
        for i in range(len(clipped_chunks[1:])):
            if clipped_chunks[i].startswith('I-'):
                if i - last_begin >= self._max_span_width:
                    clipped_chunks[i-1] = clipped_chunks[i-1].replace('I-', 'L-')
                    clipped_chunks[i] = clipped_chunks[i].replace('I-', 'B-')
            if clipped_chunks[i].startswith('B-'):
                last_begin = i
        return clipped_chunks

    def convert_bioul_to_segmental(self, chunk_tags):
        """
        Based on the specified tagging scheme, create tag sequence and instance field name.
        """
        if self.use_segmental_labels and not self.use_binary_labels:
            # Tags without BIOUL encoding.
            return [tag.split("-")[1] for tag in chunk_tags], "seg_labels"
        elif self.use_binary_labels:
            # Tags without labels.
            return [tag.split("-")[0] for tag in chunk_tags], "binary_labels"
        return chunk_tags, "labels"
