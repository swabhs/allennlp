local SEGMENTAL_LANGUAGE_MODEL = "home/swabhas/pretrained/label_encoder_seglm_transformerX2.tar.gz";
local SEGMENTAL_VOCAB = "/home/swabhas/data/language_modeling/vocab-1-billion-word-language-modeling-benchmark/";

local CHUNKER_MODEL = "/home/swabhas/pretrained/chunking_ptb_comparable.tar.gz";
local CHUNKS = "/home/swabhas/data/ner_conll2003/on-the-fly.json";

local TRAIN = "/home/swabhas/data/ner_conll2003/eng.train";
local HELDOUT = "/home/swabhas/data/ner_conll2003/eng.testa";
local GLOVE = "/home/swabhas/data/glove.6B.50d.txt";

{
  "dataset_reader": {
    "type": "conll2003",
    "tag_label": "ner",
    "coding_scheme": "BIOUL",
    "token_indexers": {
      "chunky_elmo": {
        "type": "chunky_elmo",
        "chunker_path": CHUNKER_MODEL,
        "preprocessed_chunk_file": CHUNKS,
        "segmental_vocabulary": {"directory_path": SEGMENTAL_VOCAB}
      }
    }
  },
  "train_data_path": TRAIN,
  "validation_data_path": HELDOUT,
  "model": {
    "type": "crf_tagger",
    "label_encoding": "BIOUL",
    "constrain_crf_decoding": true,
    "calculate_span_f1": true,
    "dropout": 0.5,
    "include_start_end_transitions": false,
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
        "chunky_elmo": ["character_ids", "mask", "mask_with_bos_eos", "seg_ends", "seg_map", "seg_starts", "tags"],
        "token_characters": ["token_characters"],
        "tokens": ["tokens"],
      },
      "token_embedders": {
        "chunky_elmo": {
          "type": "chunky_elmo_token_embedder",
          "segmental_path": SEGMENTAL_LANGUAGE_MODEL
        }
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": 1024,
      "hidden_size": 200,
      "num_layers": 2,
      "dropout": 0.5,
      "bidirectional": true
    },
  },
  "iterator": {
    "type": "basic",
    "batch_size": 64
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    "validation_metric": "+f1-measure-overall",
    "num_serialized_models_to_keep": 1,
    "num_epochs": 75,
    "grad_norm": 5.0,
    "patience": 25,
    "cuda_device": 0
  },
  "random_seed": std.extVar("RANDOM_SEED"),
  "numpy_seed": std.extVar("NUMPY_SEED"),
  "pytorch_seed": std.extVar("PYTORCH_SEED")
}
