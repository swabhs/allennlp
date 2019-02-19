local SEGMENTAL_LANGUAGE_MODEL = "/home/swabhas/pretrained/seglm_transformerX2.tar.gz";
local SEGMENTAL_VOCAB = "/home/swabhas/data/language_modeling/vocab-1-billion-word-language-modeling-benchmark/";

local CHUNKER_MODEL = "/home/swabhas/pretrained/chunking_ptb_comparable.tar.gz";
local CHUNKS = "/home/swabhas/data/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/data/all_chunks.json";

local TRAIN="/home/swabhas/data/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/data/train";
local HELDOUT="/home/swabhas/data/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/data/development/";

{
  "dataset_reader": {
    "type": "ontonotes_ner",
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
    "verbose_metrics": false,
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
        "chunky_elmo":{
            "type": "chunky_elmo_token_embedder",
            "segmental_path": SEGMENTAL_LANGUAGE_MODEL,
            "dropout": 0,
            "use_projection_layer": false
        },
      }
    },
    "encoder": {
      "type": "stacked_bidirectional_lstm",
      "input_size": 1024,
      "hidden_size": 200,
      "num_layers": 2,
      "recurrent_dropout_probability": 0.5,
      "use_highway": true
    },
    "feedforward": {
      "input_dim": 400,
      "num_layers": 1,
      "hidden_dims": 400,
      "activations": "tanh",
      "dropout": 0.5
    },
    "regularizer": [
      [
        "scalar_parameters",
        {
          "type": "l2",
          "alpha": 0.001
        }
      ]
    ],
    "initializer": [
      [".*tag_projection_layer.*weight", {"type": "xavier_uniform"}],
      [".*tag_projection_layer.*bias", {"type": "zero"}],
      [".*feedforward.*weight", {"type": "xavier_uniform"}],
      [".*feedforward.*bias", {"type": "zero"}],
      [".*weight_ih.*", {"type": "xavier_uniform"}],
      [".*weight_hh.*", {"type": "orthogonal"}],
      [".*bias_ih.*", {"type": "zero"}],
      [".*bias_hh.*", {"type": "lstm_hidden_bias"}]]
  },
  "iterator": {
    "type": "bucket",
    "batch_size": 64,
    "sorting_keys": [["tokens", "num_tokens"]]
  },
  "trainer": {
    "optimizer": {
        "type": "dense_sparse_adam",
        "lr": 0.001
    },
    "validation_metric": "+f1-measure-overall",
    "num_serialized_models_to_keep": 2,
    "num_epochs": 50,
    "grad_norm": 5.0,
    "patience": 8,
    "cuda_device": 0
  }
}
