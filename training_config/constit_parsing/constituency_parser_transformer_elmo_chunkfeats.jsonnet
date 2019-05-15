local TRAIN = "/home/swabhas/data/constits_ptb_predicted_pos/02-21.10way.clean";
local ALL_CHUNKS = "/home/swabhas/data/constits_ptb_predicted_pos/allennlp_output.json";
local HELDOUT = "/home/swabhas/data/constits_ptb_predicted_pos/22.auto.clean";
local TEST = "/home/swabhas/data/constits_ptb_predicted_pos/23.auto.clean";

local LM_ARCHIVE = "/home/swabhas/pretrained/log_brendan/transformer-elmo-2019.01.10.tar.gz";

{
  "dataset_reader": {
    "type": "ptb_trees",
    "use_pos_tags": true,
    "predicted_chunks": ALL_CHUNKS,
    "token_indexers": {
      "elmo": {
        "type": "elmo_characters"
      }
    }
  },
  "train_data_path": TRAIN,
  "validation_data_path": HELDOUT,
  "test_data_path": TEST,
  "model": {
    "type": "constituency_parser",
      "text_field_embedder": {
        "token_embedders": {
          "elmo": {
            "type": "bidirectional_lm_token_embedder",
            "archive_file": LM_ARCHIVE,
            "bos_eos_tokens": ["<S>", "</S>"],
            "dropout": 0.2,
            "remove_bos_eos": true,
            "requires_grad": false
          }
        }
      },
      "pos_tag_embedding": {
        "embedding_dim": 50,
        "vocab_namespace": "pos"
      },
      "chunk_tag_embedding": {
        "embedding_dim": 50,
        "vocab_namespace": "chunk"
      },
      "initializer": [
        ["tag_projection_layer.*weight", {"type": "xavier_normal"}],
        ["feedforward_layer.*weight", {"type": "xavier_normal"}],
        ["encoder._module.weight_ih.*", {"type": "xavier_normal"}],
        ["encoder._module.weight_hh.*", {"type": "orthogonal"}]
      ],
      "encoder": {
        "type": "lstm",
        "input_size": 1124,
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
    },
    "random_seed": std.extVar("RANDOM_SEED"),
    "numpy_seed": std.extVar("NUMPY_SEED"),
    "pytorch_seed": std.extVar("PYTORCH_SEED")
  }
