{
  "dataset_reader":{
    "type": "sst_tokens",
    "use_subtrees": true,
    "granularity": "5-class",
    "token_indexers": {
      "tokens": {
        "type": "single_id"
      },
      "elmo": {
        "type": "elmo_characters"
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
      "elmo": {
        "type": "elmo_characters"
      }
    }
  },
  "train_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt",
  "validation_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt",
  "test_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/test.txt",
  "model": {
    "type": "simple_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
            "type": "embedding",
            "embedding_dim": 300,
            "trainable": false
        },
        "elmo": {
          "type": "bidirectional_lm_token_embedder",
          "archive_file": "/net/nfs.corp/allennlp/swabha/pretrained/transformer-elmo-2019.01.10.tar.gz",
          "dropout": 0.0,
          "bos_eos_tokens": ["<S>", "</S>"],
          "remove_bos_eos": true,
          "requires_grad": false
        },
      }
    },
    "dropout": 0.1,
    "encoder": {
      "type": "lstm",
      "input_size": 300+1024,
      "hidden_size": 300,
      "num_layers": 2,
      "bidirectional": true
    },
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
