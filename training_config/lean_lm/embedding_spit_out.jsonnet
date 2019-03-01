local SEGMENTAL_LANGUAGE_MODEL = "/net/nfs.corp/allennlp/swabha/pretrained/seglm_transformerX2_2019-02-24.tar.gz";
local SEGMENTAL_VOCAB = "/home/swabhas/data/language_modeling/vocab-1-billion-word-language-modeling-benchmark/";

local CHUNKER_MODEL = "/home/swabhas/pretrained/chunking_ptb_comparable.tar.gz";

local TRAIN = "/home/swabhas/data/sst/dev.txt";

local GLOVE = "/home/swabhas/data/glove.840B.300d.zip";
local SPITOUT = "logtmp/chunky.json";

{
  "dataset_reader":{
    "type": "sentences",
    "token_indexers": {
      "chunky_elmo": {
        "type": "chunky_elmo",
        "chunker_path": CHUNKER_MODEL,
        "spit_out_file": SPITOUT,
        "segmental_vocabulary": {"directory_path": SEGMENTAL_VOCAB}
      }
    }
  },
  "train_data_path": TRAIN,
  "model": {
    "type": "dummy",
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
        "chunky_elmo": ["character_ids", "mask", "mask_with_bos_eos", "seg_ends", "seg_map", "seg_starts", "tags"],
        "token_characters": ["token_characters"],
        "tokens": ["tokens"],
      },
      "token_embedders": {
        "chunky_elmo":{
          "type": "chunky_elmo_token_embedder",
          "segmental_path": SEGMENTAL_LANGUAGE_MODEL,
          "dropout": 0,
          "spit_out_file": SPITOUT,
          "use_projection_layer": false
        },
      }
    },
  },
  "iterator": {
    "type": "basic",
    "batch_size" : 1
  },
  "trainer": {
    "num_serialized_models_to_keep": 2,
    "num_epochs": 1,
    "patience": 5,
    "grad_norm": 5.0,
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    }
  },
}
