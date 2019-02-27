local SEGMENTAL_LANGUAGE_MODEL = "/net/nfs.corp/allennlp/swabha/pretrained/seglm_transformerX2_2019-02-24.tar.gz";
local SEGMENTAL_VOCAB = "/home/swabhas/data/language_modeling/vocab-1-billion-word-language-modeling-benchmark/";

local CHUNKER_MODEL = "/home/swabhas/pretrained/chunking_ptb_comparable.tar.gz";
local CHUNKS = "/home/swabhas/data/sst/all_chunks.json";

local TRAIN = "/home/swabhas/data/sst/train.txt";
local HELDOUT = "/home/swabhas/data/sst/dev.txt";
local TEST = "/home/swabhas/data/sst/test.txt";

local GLOVE = "/home/swabhas/data/glove.840B.300d.zip";
{
  "dataset_reader":{
    "type": "sst_tokens",
    "use_subtrees": true,
    "granularity": "5-class",
    "token_indexers": {
      "tokens": {
        "type": "single_id"
      },
      "chunky_elmo": {
        "type": "chunky_elmo",
        "chunker_path": CHUNKER_MODEL,
        "preprocessed_chunk_file": CHUNKS,
        "segmental_vocabulary": {"directory_path": SEGMENTAL_VOCAB}
      }
    }
  },
  "validation_dataset_reader":{
    "type": "sst_tokens",
    "use_subtrees": false,
    "granularity": "5-class",
    "token_indexers": {
      "tokens": {
        "type": "single_id"
      },
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
  "test_data_path": TEST,
  "model": {
    "type": "simple_classifier",
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
        "chunky_elmo": ["character_ids", "mask", "mask_with_bos_eos", "seg_ends", "seg_map", "seg_starts", "tags"],
        "token_characters": ["token_characters"],
        "tokens": ["tokens"],
      },
      "token_embedders": {
        "tokens": {
          "pretrained_file": GLOVE,
          "type": "embedding",
          "embedding_dim": 300,
          "trainable": false
        },
        "chunky_elmo":{
          "type": "chunky_elmo_token_embedder",
          "segmental_path": SEGMENTAL_LANGUAGE_MODEL,
          "dropout": 0,
          "use_projection_layer": false
        },
      }
    },
    "dropout": 0.5,
    "pre_encode_feedforward": {
        "input_dim": 300+1024,
        "num_layers": 1,
        "hidden_dims": [300],
        "activations": ["relu"],
        "dropout": [0.25]
    },
    "encoder": {
      "type": "lstm",
      "input_size": 300,
      "hidden_size": 300,
      "num_layers": 2,
      "bidirectional": true
    },
    "integrator": {
      "type": "lstm",
      "input_size": 1800,
      "hidden_size": 300,
      "num_layers": 1,
      "bidirectional": true
    },
    "integrator_dropout": 0.1,
    "output_layer": {
        "input_dim": 2400,
        "num_layers": 3,
        "output_dims": [1200, 600, 5],
        "pool_sizes": 4,
        "dropout": [0.2, 0.3, 0.0]
    }
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size" : 100
  },
  "trainer": {
    "num_epochs": 40,
    "patience": 5,
    "grad_norm": 5.0,
    "validation_metric": "+accuracy",
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    }
  }
}
