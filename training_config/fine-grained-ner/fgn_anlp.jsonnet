local TRAIN="/home/swabhas/data/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/data/train";
local HELDOUT="/home/swabhas/data/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/data/development/";

local LM="/home/swabhas/pretrained/log_brendan/transformer-elmo-2019.01.10.tar.gz";

{
  "dataset_reader": {
    "type": "ontonotes_ner",
    "coding_scheme": "BIOUL",
    "token_indexers": {
      "elmo": {
        "type": "elmo_characters"
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
      "token_embedders": {
        "elmo": {
          "type": "bidirectional_lm_token_embedder",
          "archive_file": "/home/swabhas/pretrained/log_brendan/transformer-elmo-2019.01.10.tar.gz",
          "dropout": 0.0,
          "bos_eos_tokens": ["<S>", "</S>"],
          "remove_bos_eos": true,
          "requires_grad": false
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
