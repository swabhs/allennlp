local TRAIN = "/home/brendanr/workbenches/constituency_parsing/no_gold_pos/02-21.10way.clean";
local HELDOUT = "/home/brendanr/workbenches/constituency_parsing/no_gold_pos/22.auto.clean";
local TEST = "/home/brendanr/workbenches/constituency_parsing/no_gold_pos/23.auto.clean";


{
  "dataset_reader": {
    "type": "ptb_trees",
    "use_pos_tags": true,
    "token_indexers": {
      "chunky_elmo": {
        "type": "chunky_elmo",
        "chunker_path": "/mnt/disks/extra/workbenches/chunky/pretrained/chunking_ptb_comparable.tar.gz",
        "preprocessed_chunk_file": "/home/brendanr/workbenches/chunky/chunks_ptb_constits.json",
        "segmental_vocabulary": {
            "directory_path": "/home/brendanr/workbenches/chunky/vocabulary/"
        }
      },
      "elmo": {
        "type": "elmo_characters"
      }
    },
  },
  "train_data_path": TRAIN,
  "validation_data_path": HELDOUT,
  "test_data_path": TEST,
  "model": {
    "type": "constituency_parser",
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
        "chunky_elmo": ["character_ids","mask","mask_with_bos_eos","seg_ends","seg_map","seg_starts","tags"],
        "token_characters": ["token_characters"],
        "tokens": ["tokens"]
      },
      "token_embedders": {
        "chunky_elmo": {
            "type": "chunky_elmo_token_embedder",
            "dropout": 0.2,
            "segmental_path": "/home/swabhas/me/pretrained/seglm_transformerX2_2019-02-24.tar.gz"
        }
      }
    },
    "pos_tag_embedding": {
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
        "bidirectional": true,
        "dropout": 0.2,
        "hidden_size": 250,
        "input_size": 1074,
        "num_layers": 2
      },
      "feedforward": {
        "activations": "relu",
        "dropout": 0.1,
        "hidden_dims": 250,
        "input_dim": 500,
        "num_layers": 1
      },
      "span_extractor": {
        "type": "bidirectional_endpoint",
        "input_dim": 500
      },
    },
     "iterator": {
      "type": "bucket",
      "sorting_keys": [["tokens", "num_tokens"]],
      "batch_size": 32
    },
    "trainer": {
      "learning_rate_scheduler": {
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "mode": "max",
        "patience": 0
      },
      "num_epochs": 150,
      "num_serialized_models_to_keep": 3,
      "grad_norm": 5.0,
      "patience": 20,
      "validation_metric": "+evalb_f1_measure",
      "cuda_device": 0,
      "optimizer": {
          "type": "adam"
      },
    }
}