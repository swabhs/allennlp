local NUM_GPUS = 1;
local NUM_THREADS = 1;

local ELMO_TRANSFORMER_LM_ARCHIVE_PATH = "/net/nfs.corp/allennlp/swabha/pretrained/transformer-elmo-2019.01.10.tar.gz";
local LM_VOCAB_PATH = "/net/nfs.corp/allennlp/swabha/data/language_modeling/vocab-1-billion-word-language-modeling-benchmark/";

local TRAIN = "/net/nfs.corp/allennlp/swabha/data/language_modeling/chunks_train.conll";
local NAMESPACE = "binary_labels";


local BASE_READER = {
  "type": "segmental_conll2000",
  "use_binary_labels": true,
  "token_indexers": {
    "tokens": {
      "type": "single_id"
    },
    // "token_characters": {
    //   "type": "elmo_characters"
    // }
    "elmo": {
      "type": "elmo_characters"
    }
  },
  // "max_sequence_length": 500,
  // "start_tokens": ["<S>"],
  // "end_tokens": ["</S>"]
};

local BASE_ITERATOR = {
  "type": "bucket",
  "max_instances_in_memory": 16384 * NUM_GPUS,
  // Larger than we really desire for a batch. Since we set
  // maximum_samples_per_batch below we will pack approximately that many
  // samples in every batch.
  "batch_size": 512 * NUM_GPUS,
  "sorting_keys": [["tokens", "num_tokens"]],
  "maximum_samples_per_batch": ["num_tokens", NUM_GPUS * 2000]
};

{
  "dataset_reader":  {
    "type": "multiprocess",
    "base_reader": BASE_READER,
    "num_workers": NUM_THREADS,
    "output_queue_size": 1000
  },
  // Note: We don't set a validation_data_path because the softmax is only
  // sampled during training. Not sampling on GPUs results in a certain OOM
  // given our large vocabulary. We'll need to evaluate against the test set
  // (when we'll want a full softmax) with the CPU.
  "train_data_path": TRAIN,
  "vocabulary": {
    // Use a prespecified vocabulary for efficiency.
    "directory_path": LM_VOCAB_PATH
    // Plausible config for generating the vocabulary.
    // "tokens_to_add": {
    //     "tokens": ["<S>", "</S>"],
    //     "token_characters": ["<>/S"]
    // },
    // "min_count": {"tokens": 3}
  },
  "model": {
    "type": "segmental_language_model",
    "label_namespace": NAMESPACE,
    "bidirectional": true,
    "num_samples": 8192,
    "sparse_embeddings": true,
    "text_field_embedder": {
      // Note: This is because we only use the token_characters during
      // embedding, not the tokens themselves.
      "allow_unmatched_keys": true,
      "token_embedders": {
        "elmo":{
          "type": "bidirectional_lm_token_embedder",
          "archive_file": ELMO_TRANSFORMER_LM_ARCHIVE_PATH,
          // Dropout is kept at 0.0 because we apply dropout to the
          // projection layer before softmax.
          "dropout": 0.0,
          "bos_eos_tokens": ["<S>", "</S>"],
          "requires_grad": false
        },
      }
    },
    // Applies to the contextualized embeddings.
    "dropout": 0.1,
    // TODO(Swabha): Strictly not required, but need to implement logic that
    // allows both end2end and frozen-start settings.
    "contextualizer": {
      "type": "bidirectional_language_model_transformer",
      "input_dim": 512,
      "hidden_dim": 2048,
      "num_layers": 6,
      "dropout": 0.1,
      "input_dropout": 0.1
    },
    "forward_segmental_contextualizer": {
      "type": "bidirectional_language_model_transformer",
      "input_dim": 512,
      "hidden_dim": 2048,
      "input_dropout": 0.1,
      "num_layers": 2,
      "direction": "forward"
    },
    "backward_segmental_contextualizer": {
      "type": "bidirectional_language_model_transformer",
      "input_dim": 512,
      "hidden_dim": 2048,
      "input_dropout": 0.1,
      "num_layers": 2,
      "direction": "backward"
    },
    "softmax_projection_dim": 512,
    "label_feature_dim": 128
  },
  "iterator": {
    "type": "multiprocess",
    "base_iterator": BASE_ITERATOR,
    "num_workers": NUM_THREADS,
    // The multiprocess dataset reader and iterator use many file descriptors,
    // so we need to increase the ulimit depending on the size of this queue.
    // See https://pytorch.org/docs/stable/multiprocessing.html#file-descriptor-file-descriptor
    // for a description of the underlying issue. `ulimit -n 4096` has sufficed,
    // but that number could use tuning.
    "output_queue_size": 500
  },
  "trainer": {
    "num_epochs": 10,
    "cuda_device" : if NUM_GPUS > 1 then std.range(0, NUM_GPUS - 1) else 0,
    "model_save_interval": 7200,
    "num_serialized_models_to_keep": 2,
    "optimizer": {
      // The gradient accumulators in Adam for the running stdev and mean for
      // words not used in the sampled softmax would be decayed to zero with the
      // standard "adam" optimizer.
      "type": "dense_sparse_adam"
    },
    // TODO(brendanr): Needed with transformer too?
    // "grad_norm": 10.0,
    "learning_rate_scheduler": {
      "type": "noam",
      // See https://github.com/allenai/calypso/blob/master/calypso/train.py#L401
      "model_size": 512,
      // See https://github.com/allenai/calypso/blob/master/bin/train_transformer_lm1b.py#L51.
      // Adjusted based on our sample size relative to Calypso's.
      "warmup_steps": 6000
    }
  }
}
