// Configuration for an Bidirectional LM-augmented constituency parser based on:
//   Stern, Mitchell et al. “A Minimal Span-Based Neural Constituency Parser.” ACL (2017).
local SEGMENTAL_LANGUAGE_MODEL = "home/swabhas/pretrained/label_encoder_seglm_transformerX2.tar.gz";
local SEGMENTAL_VOCAB = "/home/swabhas/data/language_modeling/vocab-1-billion-word-language-modeling-benchmark/";

local CHUNKER_MODEL = "/home/swabhas/pretrained/chunking_ptb_comparable.tar.gz";
local CHUNKS = "/home/swabhas/data/ner_conll2003/on-the-fly.json";

local TRAIN = "/home/swabhas/data/constits_ptb_predicted_pos/02-21.10way.clean";
local HELDOOUT = "/home/swabhas/data/constits_ptb_predicted_pos/22.auto.clean";
local TEST = "/home/swabhas/data/constits_ptb_predicted_pos/23.auto.clean";

{
    "dataset_reader":{
        "type":"ptb_trees",
        "use_pos_tags": true,
        "token_indexers": {
          "elmo": {
            "type": "elmo_characters"
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
    "validation_data_path": HELDOOUT,
    "test_data_path": TEST,
    "model": {
      "type": "constituency_parser",
      "text_field_embedder": {
        "allow_unmatched_keys": true,
        "embedder_to_indexer_map": {
          "chunky_elmo": ["character_ids", "mask", "seg_ends", "seg_map", "seg_starts", "tags"],
          "token_characters": ["token_characters"],
          "tokens": ["tokens"],
        },
        "token_embedders": {
            // "elmo": {
            //   "type": "bidirectional_lm_token_embedder",
            //   "archive_file": std.extVar('BIDIRECTIONAL_LM_ARCHIVE_PATH'),
            //   "dropout": 0.2,
            //   "bos_eos_tokens": ["<S>", "</S>"],
            //   "remove_bos_eos": true,
            //   "requires_grad": true
            // }
          "chunky_elmo":{
            "type": "chunky_elmo_token_embedder",
            "segmental_path": SEGMENTAL_LANGUAGE_MODEL
          },
        }
      },
      "pos_tag_embedding":{
        "embedding_dim": 50,
        "vocab_namespace": "pos"
      },
      "initializer": [
        ["tag_projection_layer.*weight", {"type": "xavier_normal"}],
        ["feedforward_layer.*weight", {"type": "xavier_normal"}],
        ["encoder._module.weight_ih.*", {"type": "xavier_normal"}],
        ["encoder._module.weight_hh.*", {"type": "orthogonal"}]
      ],
      "encoder": {
        "type": "lstm",
        "input_size": 1074,
        "hidden_size": 250,
        "num_layers": 2,
        "bidirectional": true,
        "dropout": 0.2
      },
      "feedforward": {
        "input_dim": 500,
        "num_layers": 1,
        "hidden_dims": 250,
        "activations": "relu",
        "dropout": 0.1
      },
      "span_extractor": {
        "type": "bidirectional_endpoint",
        "input_dim": 500
      }
    },
    "iterator": {
      "type": "bucket",
      "sorting_keys": [["tokens", "num_tokens"]],
      "batch_size" : 32
    },
    "trainer": {
      "learning_rate_scheduler": {
        "type": "multi_step",
        "milestones": [40, 50, 60, 70, 80],
        "gamma": 0.8
      },
      "num_epochs": 150,
      "grad_norm": 5.0,
      "patience": 20,
      "validation_metric": "+evalb_f1_measure",
      "cuda_device": 0,
      "optimizer": {
        "type": "adadelta",
        "lr": 1.0,
        "rho": 0.95
      }
    }
  }
