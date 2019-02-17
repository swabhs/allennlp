import json
import sys
from typing import Iterable

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, OntonotesSentence, to_bioul


def _normalize_word(word: str):
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word

def _read(file_path: str, output_file: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        ontonotes_reader = Ontonotes()
        print("Reading Fine-Grained NER instances from dataset files at: {file_path}")

        with open(output_file, 'a', encoding='utf-8') as output_json:

            for sentence in _ontonotes_subset(ontonotes_reader, file_path):
                tokens = [_normalize_word(t) for t in sentence.words]

                json.dump({"words": tokens}, output_json)
                output_json.write('\n')


def _ontonotes_subset(ontonotes_reader: Ontonotes,
                          file_path: str) -> Iterable[OntonotesSentence]:
        """
        Iterates over the Ontonotes 5.0 dataset using an optional domain identifier.
        If the domain identifier is present, only examples which contain the domain
        identifier in the file path are yielded.
        """
        for conll_file in ontonotes_reader.dataset_path_iterator(file_path):
            if "/pt/" not in conll_file:
                yield from ontonotes_reader.sentence_iterator(conll_file)


if __name__ == "__main__":
    _read(sys.argv[1], "output.json")