from allennlp.common.file_utils import cached_path
from nltk.tree import Tree, ParentedTree

import json
import os
import sys

def make_input_json(file_path: str, use_subtrees: bool = True):
  output_json_file_name = os.path.join(file_path, ".json")
  with open(output_json_file_name, 'w', encoding='utf-8') as output_json:
    with open(cached_path(file_path), "r") as data_file:
      print(f"Reading instances from lines in file at: {file_path}")
      for line in data_file.readlines():
          line = line.strip("\n")
          if not line:
              continue

          parsed_line = Tree.fromstring(line)
          sentence = parsed_line.leaves()

          if use_subtrees:
            spans = get_subtree_spans(line)

          json.dump({"words": sentence, "spans": spans}, output_json)
          output_json.write('\n')
  print(f"Written to {output_json_file_name}.")

def get_subtree_spans(line: str):
  cp = ParentedTree.fromstring(line)
  for idx, pos in enumerate(cp.treepositions('leaves')):
    cp[pos] = idx

  spans = []
  for subtree in cp.subtrees():
    # sentence = subtree.leaves()
    flattened = subtree.flatten()
    span = (flattened[0], flattened[-1])
    spans.append(span)

  return spans

def combine(predicted, original):
  print(f"Reading predicted chunks for sentences in {predicted}")
  words_dict = {}
  chunks_dict = {}
  for line in open(predicted, "r"):
    cdict = json.loads(line)
    key = " ".join(cdict["words"])
    words_dict[key] = cdict["words"]
    chunks_dict[key] = cdict["tags"]

  print(f"Reading spans for sentences in {original}")
  spans_dict = {}
  for line in open(original, "r"):
    cdict = json.loads(line)
    key = " ".join(cdict["words"])
    spans_dict[key] = cdict["spans"]

  sliced_chunks_file = "/home/swabhas/data/sst/all_chunks_sliced_for_subtrees.json"
  with open(sliced_chunks_file, 'w', encoding='utf-8') as output_json:

    for key in chunks_dict:
      assert key in spans_dict and key in words_dict
      for span in spans_dict[key]:
        sentence = words_dict[key][span[0]: span[-1]+1]
        tags = chunks_dict[key][span[0]: span[-1]+1]
        json.dump({"words": sentence, "tags": tags}, output_json)
        output_json.write('\n')

  print(f"Sliced chunks for trees in {sliced_chunks_file}")


if __name__ == "__main__":
  # make_input_json(sys.argv[1], True)
  predicted = "/home/swabhas/data/sst/all_spans_chunks.json"
  original = "/home/swabhas/data/sst/all_spans.json"

  combine(predicted, original)