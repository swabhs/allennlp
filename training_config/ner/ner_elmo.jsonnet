// Configuration for the NER model with ELMo, modified slightly from
// the version included in "Deep Contextualized Word Representations",
// (https://arxiv.org/abs/1802.05365).  Compared to the version in this paper,
// this configuration replaces the original Senna word embeddings with
// 50d GloVe embeddings.
//
// There is a trained model available at https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.04.30.tar.gz
// with test set F1 of 92.51 compared to the single model reported
// result of 92.22 +/- 0.10.
local TRAIN = "/home/swabhas/data/ner_conll2003/eng.train";
local HELDOUT = "/home/swabhas/data/ner_conll2003/eng.testa";
local GLOVE = "/home/swabhas/data/glove.6B.50d.txt";
{
  "dataset_reader": {
    "type": "conll2003",
    "tag_label": "ner",
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
    "constrain_crf_decoding": true,
    "calculate_span_f1": true,
    "dropout": 0.5,
    "include_start_end_transitions": false,
    "text_field_embedder": {
      "token_embedders": {
        "elmo":{
            "type": "elmo_token_embedder",
            "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
            "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
            "do_layer_norm": false,
            "dropout": 0.0
        }
      },
    },
    "encoder": {
        "type": "lstm",
        "input_size": 1024,
        "hidden_size": 200,
        "num_layers": 2,
        "dropout": 0.5,
        "bidirectional": true
    },
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
    "num_serialized_models_to_keep": 1,
    "num_epochs": 75,
    "grad_norm": 5.0,
    "patience": 25,
    "cuda_device": 0
  },
  "random_seed": std.extVar("RANDOM_SEED"),
  "numpy_seed": std.extVar("NUMPY_SEED"),
  "pytorch_seed": std.extVar("PYTORCH_SEED")
}
