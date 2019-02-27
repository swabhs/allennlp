from allennlp.common.file_utils import cached_path
from nltk.tree import Tree

import json
import sys

def read(file_path: str, use_subtrees: bool = True):
  with open(file_path+".json", 'w', encoding='utf-8') as output_json:
    with open(cached_path(file_path), "r") as data_file:
      print(f"Reading instances from lines in file at: {file_path}")
      for line in data_file.readlines():
          line = line.strip("\n")
          if not line:
              continue
          parsed_line = Tree.fromstring(line)
          if use_subtrees:
              for subtree in parsed_line.subtrees():
                  sentence = subtree.leaves()
          else:
              sentence = parsed_line.leaves()
          json.dump({"sentence": sentence}, output_json)
          output_json.write('\n')

if __name__ == "__main__":
  read(sys.argv[1])