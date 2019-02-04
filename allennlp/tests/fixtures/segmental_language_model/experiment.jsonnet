{
  "dataset_reader":  {
    "type": "segmental_conll2000",
    "token_indexers": {
      "tokens": {
        "type": "single_id"
      },
      "token_characters": {
        "type": "elmo_characters"
      },
    }
  },
  "train_data_path": "allennlp/tests/fixtures/data/chunks_bioul.conll",
  "validation_data_path": "allennlp/tests/fixtures/data/chunks_bioul_dev.conll",
  "vocabulary": {
      "tokens_to_add": {
          "tokens": ["<S>", "</S>"],
          "token_characters": ["<>/S"]
      },
  },
  "model": {
    "type": "segmental_language_model",
    "bidirectional": true,
    "num_samples": 12,
    "sparse_embeddings": true,
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "token_embedders": {
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
                "num_embeddings": 262,
                "embedding_dim": 16
            },
            "encoder": {
                "type": "cnn-highway",
                "activation": "relu",
                "embedding_dim": 16,
                "filters": [
                    [1, 32],
                    [2, 32],
                    [3, 64]],
                "num_highway": 2,
                "projection_dim": 8,
                "projection_location": "after_highway",
                "do_layer_norm": true
            }
        }
      }
    },
    "dropout": 0.1,
    "contextualizer": {
        "type": "bidirectional_language_model_transformer",
        "input_dim": 8,
        "hidden_dim": 16,
        "num_layers": 6,
        "dropout": 0.1,
        "input_dropout": 0.1
    },
    "forward_segmental_contextualizer": {
      "type": "bidirectional_language_model_transformer",
      "input_dim": 8,
      "hidden_dim": 16,
      "input_dropout": 0.1,
      "num_layers": 2,
      "direction": "forward"
    },
    "backward_segmental_contextualizer": {
      "type": "bidirectional_language_model_transformer",
      "input_dim": 8,
      "hidden_dim": 16,
      "input_dropout": 0.1,
      "num_layers": 2,
      "direction": "backward"
    },
    "softmax_projection_dim": 13,
    "label_feature_dim": 14
  },
  "iterator": {
    "type": "basic",
    "batch_size": 32,
  },
  "trainer": {
    "num_epochs": 3,
    "cuda_device" : -1,
    "optimizer": {
      "type": "sgd",
      "lr": 0.01
    },
    "log_batch_size_period": 1,
  }
}
