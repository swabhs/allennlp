
from typing import Dict, List, Tuple
import json
import logging
import os

from overrides import overrides
# NLTK is so performance orientated (ha ha) that they have lazy imports. Why? Who knows.
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader # pylint: disable=no-name-in-module
from nltk.tree import ParentedTree, Tree

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import bioul_tags_to_spans
from allennlp.data.fields import TextField, SpanField, SequenceLabelField, ListField, MetadataField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_utils.span_utils import enumerate_spans
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("ptb_trees")
class PennTreeBankConstituencySpanDatasetReader(DatasetReader):
    """
    Reads constituency parses from the WSJ part of the Penn Tree Bank from the LDC.
    This ``DatasetReader`` is designed for use with a span labelling model, so
    it enumerates all possible spans in the sentence and returns them, along with gold
    labels for the relevant spans present in a gold tree, if provided.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    use_pos_tags : ``bool``, optional, (default = ``True``)
        Whether or not the instance should contain gold POS tags
        as a field.
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be consumed lazily.
    label_namespace_prefix : ``str``, optional, (default = ``""``)
        Prefix used for the label namespace.  The ``span_labels`` will use
        namespace ``label_namespace_prefix + 'labels'``, and if using POS
        tags their namespace is ``label_namespace_prefix + pos_label_namespace``.
    pos_label_namespace : ``str``, optional, (default = ``"pos"``)
        The POS tag namespace is ``label_namespace_prefix + pos_label_namespace``.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 use_pos_tags: bool = True,
                 lazy: bool = False,
                 label_namespace_prefix: str = "",
                 pos_label_namespace: str = "pos",
                 chunk_label_namespace: str = "chunk",
                 remove_bioul_tags: bool = False,
                 predicted_chunks: str = None) -> None:
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._use_pos_tags = use_pos_tags
        self._label_namespace_prefix = label_namespace_prefix
        self._pos_label_namespace = pos_label_namespace

        self._chunk_label_namespace = chunk_label_namespace
        self._remove_bioul_tags = remove_bioul_tags
        self._sents_chunks: Dict[str, List[str]] = {}
        if predicted_chunks is not None:
            self._read_predicted_chunks(predicted_chunks)

        self.overlap_chunks = 0
        self.total_chunks = 0

    def _read_predicted_chunks(self, chunks_json_file_path:str) -> None:
        chunks_json_file_path = cached_path(chunks_json_file_path)
        logger.info("Reading chunks from json lines at: %s", chunks_json_file_path)

        with open(chunks_json_file_path) as predicted_chunks_json:
            for line in predicted_chunks_json:
                prediction = json.loads(line)
                sentence = prediction["words"]
                tags = prediction["tags"]
                self._sents_chunks[' '.join(sentence)] = tags

    @overrides
    def _read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        directory, filename = os.path.split(file_path)
        logger.info("Reading instances from lines in file at: %s", file_path)
        for parse in BracketParseCorpusReader(root=directory, fileids=[filename]).parsed_sents():

            self._strip_functional_tags(parse)
            # This is un-needed and clutters the label space.
            # All the trees also contain a root S node.
            if parse.label() == "VROOT":
                parse = parse[0]
            pos_tags = [x[1] for x in parse.pos()] if self._use_pos_tags else None

            sent = ' '.join(parse.leaves())
            chunk_tags = None
            if self._sents_chunks:
                if sent in self._sents_chunks:
                    chunk_tags = self._sents_chunks[sent]
                    assert len(chunk_tags) == len(parse.leaves())
                    self._analyze_overlap(chunk_tags, parse)
                    if self._remove_bioul_tags:
                        chunk_tags = [t.split("-")[-1] for t in chunk_tags]
                else:
                    raise ConfigurationError("sentence absent from corpus!", sent)

            yield self.text_to_instance(parse.leaves(), pos_tags, parse, chunk_tags)
        # if self._sents_chunks:
        #     print(self.overlap_chunks/self.total_chunks)

    @overrides
    def text_to_instance(self, # type: ignore
                         tokens: List[str],
                         pos_tags: List[str] = None,
                         gold_tree: Tree = None,
                         chunk_tags: List[str] = None) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.

        Parameters
        ----------
        tokens : ``List[str]``, required.
            The tokens in a given sentence.
        pos_tags ``List[str]``, optional, (default = None).
            The POS tags for the words in the sentence.
        gold_tree : ``Tree``, optional (default = None).
            The gold parse tree to create span labels from.

        Returns
        -------
        An ``Instance`` containing the following fields:
            tokens : ``TextField``
                The tokens in the sentence.
            pos_tags : ``SequenceLabelField``
                The POS tags of the words in the sentence.
                Only returned if ``use_pos_tags`` is ``True``
            spans : ``ListField[SpanField]``
                A ListField containing all possible subspans of the
                sentence.
            span_labels : ``SequenceLabelField``, optional.
                The constituency tags for each of the possible spans, with
                respect to a gold parse tree. If a span is not contained
                within the tree, a span will have a ``NO-LABEL`` label.
            gold_tree : ``MetadataField(Tree)``
                The gold NLTK parse tree for use in evaluation.
        """
        # pylint: disable=arguments-differ
        text_field = TextField([Token(x) for x in tokens], token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}

        pos_namespace = self._label_namespace_prefix + self._pos_label_namespace
        if self._use_pos_tags and pos_tags is not None:
            pos_tag_field = SequenceLabelField(pos_tags, text_field,
                                               label_namespace=pos_namespace)
            fields["pos_tags"] = pos_tag_field
        elif self._use_pos_tags:
            raise ConfigurationError("use_pos_tags was set to True but no gold pos"
                                     " tags were passed to the dataset reader.")

        chunk_namespace = self._label_namespace_prefix + self._chunk_label_namespace
        if chunk_tags is not None:
            chunk_tag_field = SequenceLabelField(chunk_tags, text_field,
                                                 label_namespace=chunk_namespace)
            fields["chunk_tags"] = chunk_tag_field

        spans: List[Field] = []
        gold_labels = []

        if gold_tree is not None:
            gold_spans: Dict[Tuple[int, int], str] = {}
            self._get_gold_spans(gold_tree, 0, gold_spans)

        else:
            gold_spans = None
        for start, end in enumerate_spans(tokens):
            spans.append(SpanField(start, end, text_field))

            if gold_spans is not None:
                if (start, end) in gold_spans.keys():
                    gold_labels.append(gold_spans[(start, end)])
                else:
                    gold_labels.append("NO-LABEL")

        metadata = {"tokens": tokens}
        if gold_tree:
            metadata["gold_tree"] = gold_tree
        if self._use_pos_tags:
            metadata["pos_tags"] = pos_tags

        fields["metadata"] = MetadataField(metadata)

        span_list_field: ListField = ListField(spans)
        fields["spans"] = span_list_field
        if gold_tree is not None:
            fields["span_labels"] = SequenceLabelField(gold_labels,
                                                       span_list_field,
                                                       label_namespace=self._label_namespace_prefix + "labels")
        return Instance(fields)

    def _strip_functional_tags(self, tree: Tree) -> None:
        """
        Removes all functional tags from constituency labels in an NLTK tree.
        We also strip off anything after a =, - or | character, because these
        are functional tags which we don't want to use.

        This modification is done in-place.
        """
        clean_label = tree.label().split("=")[0].split("-")[0].split("|")[0]
        tree.set_label(clean_label)
        for child in tree:
            if not isinstance(child[0], str):
                self._strip_functional_tags(child)

    def _get_gold_spans(self, # pylint: disable=arguments-differ
                        tree: Tree,
                        index: int,
                        typed_spans: Dict[Tuple[int, int], str]) -> int:
        """
        Recursively construct the gold spans from an nltk ``Tree``.
        Labels are the constituents, and in the case of nested constituents
        with the same spans, labels are concatenated in parent-child order.
        For example, ``(S (NP (D the) (N man)))`` would have an ``S-NP`` label
        for the outer span, as it has both ``S`` and ``NP`` label.
        Spans are inclusive.

        TODO(Mark): If we encounter a gold nested labelling at test time
        which we haven't encountered, we won't be able to run the model
        at all.

        Parameters
        ----------
        tree : ``Tree``, required.
            An NLTK parse tree to extract spans from.
        index : ``int``, required.
            The index of the current span in the sentence being considered.
        typed_spans : ``Dict[Tuple[int, int], str]``, required.
            A dictionary mapping spans to span labels.

        Returns
        -------
        typed_spans : ``Dict[Tuple[int, int], str]``.
            A dictionary mapping all subtree spans in the parse tree
            to their constituency labels. POS tags are ignored.
        """
        # NLTK leaves are strings.
        if isinstance(tree[0], str):
            # The "length" of a tree is defined by
            # NLTK as the number of children.
            # We don't actually want the spans for leaves, because
            # their labels are POS tags. Instead, we just add the length
            # of the word to the end index as we iterate through.
            end = index + len(tree)
        else:
            # otherwise, the tree has children.
            child_start = index
            for child in tree:
                # typed_spans is being updated inplace.
                end = self._get_gold_spans(child, child_start, typed_spans)
                child_start = end
            # Set the end index of the current span to
            # the last appended index - 1, as the span is inclusive.
            span = (index, end - 1)
            current_span_label = typed_spans.get(span)
            if current_span_label is None:
                # This span doesn't have nested labels, just
                # use the current node's label.
                typed_spans[span] = tree.label()
            else:
                # This span has already been added, so prepend
                # this label (as we are traversing the tree from
                # the bottom up).
                typed_spans[span] = tree.label() + "-" + current_span_label

        return end

    def _analyze_overlap(self, chunk_tags: List[str], parse: Tree):
        parse = ParentedTree.fromstring(str(parse))
        # Replace leaves with node-ids.
        idx = 0
        for position in parse.treepositions('leaves'):
            parse[position] = idx
            idx += 1

        constit_spans : Dict[Tuple[int, int], str] = {}
        for subtree in parse.subtrees():
            flat_subtree = subtree.flatten()
            span = (flat_subtree[0], flat_subtree[-1])
            if span not in constit_spans:
                constit_spans[span] = []
            constit_spans[span].append(flat_subtree.label())

        spans = bioul_tags_to_spans(chunk_tags)
        # print(spans, constit_spans)

        self.overlap_chunks +=  sum([s[1] in constit_spans and s[0] in constit_spans[s[1]] for s in spans])
        self.total_chunks += len(spans)
