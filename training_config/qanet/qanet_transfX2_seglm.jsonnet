
local SEGMENTAL_LANGUAGE_MODEL = "/home/swabhas/me/pretrained/seglm_transformerX2_2019-02-24.tar.gz";
local SEGMENTAL_VOCAB = "/home/swabhas/me/data/language_modeling/vocab-1-billion-word-language-modeling-benchmark/";

local CHUNKER_MODEL = "/home/swabhas/me/pretrained/chunking_ptb_comparable.tar.gz";
local CHUNKS = "/home/swabhas/me/data/squad/all_chunks.json";

local TRAIN = "/home/swabhas/me/data/squad/squad-train-v1.1.json";
local DEV = "/home/swabhas/me/data/squad/squad-dev-v1.1.json";

local GLOVE = "/home/swabhas/me/data/glove.840B.300d.lower.converted.zip";

{
    "dataset_reader": {
        "type": "squad",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            },
            "token_characters": {
                "type": "characters",
                "min_padding_length": 5
            },
            "chunky_elmo": {
                "type": "chunky_elmo",
                "chunker_path": CHUNKER_MODEL,
                "preprocessed_chunk_file": CHUNKS,
                "segmental_vocabulary": {"directory_path": SEGMENTAL_VOCAB}
            }
        },
        "passage_length_limit": 400,
        "question_length_limit": 50,
        "skip_invalid_examples": true
    },
    "validation_dataset_reader": {
        "type": "squad",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            },
            "token_characters": {
                "type": "characters",
                "min_padding_length": 5
            },
           "chunky_elmo": {
                "type": "chunky_elmo",
                "chunker_path": CHUNKER_MODEL,
                "preprocessed_chunk_file": CHUNKS,
                "segmental_vocabulary": {"directory_path": SEGMENTAL_VOCAB}
            }
        },
        "passage_length_limit": 1000,
        "question_length_limit": 100,
        "skip_invalid_examples": false
    },
    "vocabulary": {
        "min_count": {
            "token_characters": 200
        },
        "pretrained_files": {
            // This embedding file is created from the Glove 840B 300d embedding file.
            // We kept all the original lowercased words and their embeddings. But there are also many words
            // with only the uppercased version. To include as many words as possible, we lowered those words
            // and used the embeddings of uppercased words as an alternative.
            "tokens": GLOVE
        },
        "only_include_pretrained_words": true
    },
    "train_data_path": TRAIN,
    "validation_data_path": DEV,
    "model": {
        "type": "qanet",
        "text_field_embedder": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "chunky_elmo": ["character_ids", "mask", "mask_with_bos_eos", "seg_ends", "seg_map", "seg_starts", "tags"],
                "token_characters": ["token_characters"],
                "tokens": ["tokens"],
            },
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "pretrained_file": GLOVE,
                    "embedding_dim": 300,
                    "trainable": false
                },
                "token_characters": {
                    "type": "character_encoding",
                    "embedding": {
                        "embedding_dim": 64
                    },
                    "encoder": {
                        "type": "cnn",
                        "embedding_dim": 64,
                        "num_filters": 200,
                        "ngram_filter_sizes": [
                            5
                        ]
                    }
                },
                "chunky_elmo":{
                    "type": "chunky_elmo_token_embedder",
                    "segmental_path": SEGMENTAL_LANGUAGE_MODEL,
                    "dropout": 0.4
                },
            }
        },
        "num_highway_layers": 2,
        "phrase_layer": {
            "type": "qanet_encoder",
            "attention_dropout_prob": 0,
            "attention_projection_dim": 128,
            "conv_kernel_size": 7,
            "dropout_prob": 0.1,
            "feedforward_hidden_dim": 128,
            "hidden_dim": 128,
            "input_dim": 128,
            "layer_dropout_undecayed_prob": 0.1,
            "num_blocks": 1,
            "num_convs_per_block": 4,
            "num_attention_heads": 8,
        },
        "matrix_attention_layer": {
            "type": "linear",
            "tensor_1_dim": 128,
            "tensor_2_dim": 128,
            "combination": "x,y,x*y"
        },
        "modeling_layer": {
            "type": "qanet_encoder",
            "attention_dropout_prob": 0,
            "attention_projection_dim": 128,
            "conv_kernel_size": 5,
            "dropout_prob": 0.1,
            "feedforward_hidden_dim": 128,
            "hidden_dim": 128,
            "input_dim": 128,
            "layer_dropout_undecayed_prob": 0.1,
            "num_blocks": 6,
            "num_convs_per_block": 2,
            "num_attention_heads": 8,
        },
        "dropout_prob": 0.1,
        "regularizer": [
            [
                ".*",
                {
                    "type": "l2",
                    "alpha": 1e-07
                }
            ]
        ]
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [
            [
                "passage",
                "num_tokens"
            ],
            [
                "question",
                "num_tokens"
            ]
        ],
        "batch_size": 16,
        "max_instances_in_memory": 600
    },
    "trainer": {
        "num_epochs": 50,
        "num_serialized_models_to_keep": 3,
        "grad_norm": 5,
        "patience": 10,
        "validation_metric": "+em",
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 0.001,
            "betas": [
                0.8,
                0.999
            ],
            "eps": 1e-07
        },
        // "moving_average": {
        //     "type": "exponential",
        //     "decay": 0.9999
        // }
    }
}
