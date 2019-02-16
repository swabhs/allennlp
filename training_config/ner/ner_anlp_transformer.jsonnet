local ELMO_TRANSFORMER_LM = "/home/swabhas/pretrained/log_brendan/transformer-elmo-2019.01.10.tar.gz";

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
        "elmo": {
          "type": "bidirectional_lm_token_embedder",
          "archive_file": ELMO_TRANSFORMER_LM,
          "dropout": 0.0,
          "bos_eos_tokens": ["<S>", "</S>"],
          "remove_bos_eos": true,
          "requires_grad": false
        },
      }
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
  }
}
