{
  "dataset_reader": {
    "type": "ontonotes_ner",
    "coding_scheme": "BIOUL",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "token_characters": {
        "type": "characters"
      },
      "elmo": {
        "type": "elmo_characters"
     }
    }
  },
  "train_data_path":"/ontonotes/train",
  "validation_data_path": "/ontonotes/development",
  "model": {
    "type": "crf_tagger",
    "constraint_type": "BIOUL",
    "verbose_metrics": true,
    "dropout": 0.5,
    "include_start_end_transitions": false,
    "text_field_embedder": {
      "tokens": {
        "type": "embedding",
        "embedding_dim": 50,
        "pretrained_file": "/glove/glove.6B.50d.txt.gz",
        "sparse": true,
        "trainable": true
      },
      "elmo":{
        "type": "elmo_token_embedder",
        "weight_file": "/elmo/elmo_2x4096_512_2048cnn_2xhighway_enwiki_news_967500_weights.hdf5",
        "options_file": "/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json",
        "do_layer_norm": false,
        "dropout": 0.0
      },
      "token_characters": {
        "type": "character_encoding",
        "embedding": {
          "embedding_dim": 25,
          "sparse": true
        },
        "encoder": {
          "type": "lstm",
          "input_size": 25,
          "hidden_size": 128,
          "num_layers": 1
        }
      }
    },
    "encoder": {
      "type": "stacked_bidirectional_lstm",
      "input_size": 1202,
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
    "num_serialized_models_to_keep": 3,
    "num_epochs": 30,
    "grad_norm": 5.0,
    "patience": 25,
    "cuda_device": 0
  }
}
