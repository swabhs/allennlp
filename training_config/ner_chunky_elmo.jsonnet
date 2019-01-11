//
{

  "dataset_reader": {
    "type": "conll2003",
    "tag_label": "ner",
    "coding_scheme": "IOB1",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 3
      },
      "chunky_elmo": {
        "type": "elmo_characters"
     }
    }
  },
  "train_data_path": "/home/swabhas/data/ner_conll2003/bio/train.txt",
  "validation_data_path": "/home/swabhas/data/ner_conll2003/bio/valid.txt",
  "model": {
    "type": "crf_tagger",
    "constraint_type": "BIOUL",
    "dropout": 0.5,
    "include_start_end_transitions": false,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 50,
            "pretrained_file": "/home/swabhas/data/glove.6B.50d.zip",
            "trainable": true
        },
        "chunky_elmo":{
            "type": "chunky_elmo_token_embedder",
            "chunker_path": "/home/swabhas/pretrained/log_chunking_ptb_comparable/model.tar.gz",
            "segmental_lm_path": "/home/swabhas/pretrained/log_1b_bilm_anlptransformer/model.tar.gz",
        },
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
            "embedding_dim": 16
            },
            "encoder": {
            "type": "cnn",
            "embedding_dim": 16,
            "num_filters": 128,
            "ngram_filter_sizes": [3],
            "conv_layer_activation": "relu"
            }
        }
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": 1202,
      "hidden_size": 200,
      "num_layers": 2,
      "dropout": 0.5,
      "bidirectional": true
    },
    "regularizer": [
      [
        "scalar_parameters",
        {
          "type": "l2",
          "alpha": 0.1
        }
      ]
    ]
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
    "num_serialized_models_to_keep": 3,
    "num_epochs": 75,
    "grad_norm": 5.0,
    "patience": 25,
    "cuda_device": 0
  }
}
