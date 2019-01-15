import json
import logging
from typing import Union, List, Dict, Any
import warnings

import torch
from torch.nn.modules import Dropout

import numpy
# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category=FutureWarning)
#     import h5py
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.checks import ConfigurationError
from allennlp.common import Params
from allennlp.common.util import lazy_groups_of
from allennlp.modules.elmo import _ElmoCharacterEncoder
from allennlp.modules.highway import Highway
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.seq2seq_encoders.bidirectional_language_model_transformer import BidirectionalLanguageModelTransformer
from allennlp.nn.util import remove_sentence_boundaries, add_sentence_boundary_token_ids, get_device_of, device_mapping
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper, ELMoTokenCharactersIndexer
from allennlp.data.dataset import Batch
from allennlp.data import Token, Vocabulary, Instance
from allennlp.data.fields import TextField


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# pylint: disable=attribute-defined-outside-init


class SegmentalElmo(torch.nn.Module):
    """
    Compute ELMo representations using a pre-trained bidirectional language model.

    See "Deep contextualized word representations", Peters et al. for details.

    This module takes character id input and computes ``num_output_representations`` different layers
    of ELMo representations.  Typically ``num_output_representations`` is 1 or 2.  For example, in
    the case of the SRL model in the above paper, ``num_output_representations=1`` where ELMo was included at
    the input token representation layer.  In the case of the SQuAD model, ``num_output_representations=2``
    as ELMo was also included at the GRU output layer.

    In the implementation below, we learn separate scalar weights for each output layer,
    but only run the biLM once on each input sequence for efficiency.

    Parameters
    ----------
    options_file : ``str``, required.
        ELMo JSON options file
    weight_file : ``str``, required.
        ELMo hdf5 weight file
    num_output_representations: ``int``, required.
        The number of ELMo representation to output with
        different linear weighted combination of the 3 layers (i.e.,
        character-convnet output, 1st lstm output, 2nd lstm output).
    requires_grad: ``bool``, optional
        If True, compute gradient of ELMo parameters for fine tuning.
    do_layer_norm : ``bool``, optional, (default = False).
        Should we apply layer normalization (passed to ``ScalarMix``)?
    dropout : ``float``, optional, (default = 0.5).
        The dropout to be applied to the ELMo representations.
    vocab_to_cache : ``List[str]``, optional, (default = None).
        A list of words to pre-compute and cache character convolutions
        for. If you use this option, Elmo expects that you pass word
        indices of shape (batch_size, timesteps) to forward, instead
        of character indices. If you use this option and pass a word which
        wasn't pre-cached, this will break.
    keep_sentence_boundaries : ``bool``, optional, (default = False)
        If True, the representation of the sentence boundary tokens are
        not removed.
    scalar_mix_parameters : ``List[int]``, optional, (default = None)
        If not ``None``, use these scalar mix parameters to weight the representations
        produced by different layers. These mixing weights are not updated during
        training.
    module : ``torch.nn.Module``, optional, (default = None).
        If provided, then use this module instead of the pre-trained ELMo biLM.
        If using this option, then pass ``None`` for both ``options_file``
        and ``weight_file``.  The module must provide a public attribute
        ``num_layers`` with the number of internal layers and its ``forward``
        method must return a ``dict`` with ``activations`` and ``mask`` keys
        (see `_ElmoBilm`` for an example).  Note that ``requires_grad`` is also
        ignored with this option.
    """
    def __init__(self,
                 options_file: str,
                 weight_file: str,
                 num_output_representations: int,
                 requires_grad: bool = False,
                 do_layer_norm: bool = False,
                 dropout: float = 0.5,
                 vocab_to_cache: List[str] = None,
                 keep_sentence_boundaries: bool = False,
                 scalar_mix_parameters: List[float] = None,
                 module: torch.nn.Module = None) -> None:
        super(SegmentalElmo, self).__init__()

        logger.info("Initializing ELMo")
        if module is not None:
            if options_file is not None or weight_file is not None:
                raise ConfigurationError(
                        "Don't provide options_file or weight_file with module")
            self._elmo_lstm = module
        else:
            self._elmo_lstm = _BiLMTransformerWrapper(options_file,
                                                      weight_file,
                                                      requires_grad=requires_grad,
                                                      vocab_to_cache=vocab_to_cache)
            self._seg_encoder = _BiLMTransformerWrapper(options_file,
                                                      weight_file,
                                                      requires_grad=requires_grad,
                                                      vocab_to_cache=vocab_to_cache,
                                                      is_segmental=True)
        self._has_cached_vocab = vocab_to_cache is not None
        self._keep_sentence_boundaries = keep_sentence_boundaries
        self._dropout = Dropout(p=dropout)
        self._scalar_mixes: Any = []
        for k in range(num_output_representations):
            scalar_mix = ScalarMix(
                    self._elmo_lstm.num_layers,
                    do_layer_norm=do_layer_norm,
                    initial_scalar_parameters=scalar_mix_parameters,
                    trainable=scalar_mix_parameters is None)
            self.add_module('scalar_mix_{}'.format(k), scalar_mix)
            self._scalar_mixes.append(scalar_mix)

    def get_output_dim(self):
        # TODO(swabha): check?
        return self._seg_encoder.get_output_dim()

    def forward(self,    # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                tags: torch.Tensor,
                word_inputs: torch.Tensor = None) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Parameters
        ----------
        inputs: ``torch.Tensor``, required.
        Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.
        word_inputs : ``torch.Tensor``, required.
            If you passed a cached vocab, you can in addition pass a tensor of shape
            ``(batch_size, timesteps)``, which represent word ids which have been pre-cached.

        Returns
        -------
        Dict with keys:
        ``'elmo_representations'``: ``List[torch.Tensor]``
            A ``num_output_representations`` list of ELMo representations for the input sequence.
            Each representation is shape ``(batch_size, timesteps, embedding_dim)``
        ``'mask'``:  ``torch.Tensor``
            Shape ``(batch_size, timesteps)`` long tensor with sequence mask.
        """
        # reshape the input if needed
        original_shape = inputs.size()
        if len(original_shape) > 3:
            timesteps, num_characters = original_shape[-2:]
            reshaped_inputs = inputs.view(-1, timesteps, num_characters)
        else:
            reshaped_inputs = inputs

        if word_inputs is not None:
            original_word_size = word_inputs.size()
            if self._has_cached_vocab and len(original_word_size) > 2:
                reshaped_word_inputs = word_inputs.view(-1, original_word_size[-1])
            elif not self._has_cached_vocab:
                logger.warning("Word inputs were passed to ELMo but it does not have a cached vocab.")
                reshaped_word_inputs = None
            else:
                reshaped_word_inputs = word_inputs
        else:
            reshaped_word_inputs = word_inputs

        # run the biLM
        bilm_output = self._elmo_lstm(reshaped_inputs, reshaped_word_inputs)

        # TODO(swabha): add segmental logic.

        layer_activations = bilm_output['activations']
        mask_with_bos_eos = bilm_output['mask']

        # compute the elmo representations
        representations = []
        for i in range(len(self._scalar_mixes)):
            scalar_mix = getattr(self, 'scalar_mix_{}'.format(i))
            representation_with_bos_eos = scalar_mix(layer_activations, mask_with_bos_eos)
            if self._keep_sentence_boundaries:
                processed_representation = representation_with_bos_eos
                processed_mask = mask_with_bos_eos
            else:
                representation_without_bos_eos, mask_without_bos_eos = remove_sentence_boundaries(
                        representation_with_bos_eos, mask_with_bos_eos)
                processed_representation = representation_without_bos_eos
                processed_mask = mask_without_bos_eos
            representations.append(self._dropout(processed_representation))

        # reshape if necessary
        if word_inputs is not None and len(original_word_size) > 2:
            mask = processed_mask.view(original_word_size)
            elmo_representations = [representation.view(original_word_size + (-1, ))
                                    for representation in representations]
        elif len(original_shape) > 3:
            mask = processed_mask.view(original_shape[:-1])
            elmo_representations = [representation.view(original_shape[:-1] + (-1, ))
                                    for representation in representations]
        else:
            mask = processed_mask
            elmo_representations = representations

        return {'elmo_representations': elmo_representations, 'mask': mask}

    # The add_to_archive logic here requires a custom from_params.
    @classmethod
    def from_params(cls, params: Params) -> 'SegmentalElmo':
        # Add files to archive
        params.add_file_to_archive('options_file')
        params.add_file_to_archive('weight_file')

        options_file = params.pop('options_file')
        weight_file = params.pop('weight_file')
        requires_grad = params.pop('requires_grad', False)
        num_output_representations = params.pop('num_output_representations')
        do_layer_norm = params.pop_bool('do_layer_norm', False)
        keep_sentence_boundaries = params.pop_bool('keep_sentence_boundaries', False)
        dropout = params.pop_float('dropout', 0.5)
        scalar_mix_parameters = params.pop('scalar_mix_parameters', None)
        params.assert_empty(cls.__name__)

        return cls(options_file=options_file,
                   weight_file=weight_file,
                   num_output_representations=num_output_representations,
                   requires_grad=requires_grad,
                   do_layer_norm=do_layer_norm,
                   keep_sentence_boundaries=keep_sentence_boundaries,
                   dropout=dropout,
                   scalar_mix_parameters=scalar_mix_parameters)


class _BiLMTransformerWrapper(torch.nn.Module):
    def __init__(self,
                 options_file: str,
                 weight_file: str,
                 requires_grad: bool = False,
                 vocab_to_cache: List[str] = None,
                 is_segmental: bool = False) -> None:
        super(_BiLMTransformerWrapper, self).__init__()

        self._token_embedder = _ElmoCharacterEncoder(options_file, weight_file, requires_grad=requires_grad)

        self._requires_grad = requires_grad
        if requires_grad and vocab_to_cache:
            logging.warning("You are fine tuning ELMo and caching char CNN word vectors. "
                            "This behaviour is not guaranteed to be well defined, particularly. "
                            "if not all of your inputs will occur in the vocabulary cache.")
        # This is an embedding, used to look up cached
        # word vectors built from character level cnn embeddings.
        self._word_embedding = None
        self._bos_embedding: torch.Tensor = None
        self._eos_embedding: torch.Tensor = None
        if vocab_to_cache:
            logging.info("Caching character cnn layers for words in vocabulary.")
            # This sets 3 attributes, _word_embedding, _bos_embedding and _eos_embedding.
            # They are set in the method so they can be accessed from outside the
            # constructor.
            self.create_cached_cnn_embeddings(vocab_to_cache)

        with open(cached_path(options_file), 'r') as fin:
            options = json.load(fin)

        # TODO(swabha): This won't work unless the config is modified.
        if is_segmental:
            if 'segmental_encoder' not in options or 'transformer' not in options['segmental_encoder']:
                raise ConfigurationError("This module only supports Transformer segmental encoders.")
            self._transformer = BidirectionalLanguageModelTransformer(input_dim=options['segmental_encoder']['transformer']['input_dim'],
                                                                  hidden_dim=options['segmental_encoder']['transformer']['hidden_dim'],
                                                                  num_layers=options['segmental_encoder']['transformer']['num_layers'],
                                                                  input_dropout=options['segmental_encoder']['transformer']['input_dropout'])
            self.num_layers = options['segmental_encoder']['transformer']['num_layers'] + 1
        else:
            if 'transformer' not in options:
                raise ConfigurationError("This module only supports Transformer encoders.")
            self._transformer = BidirectionalLanguageModelTransformer(input_dim=options['transformer']['input_dim'],
                                                                    hidden_dim=options['transformer']['hidden_dim'],
                                                                    num_layers=options['transformer']['num_layers'],
                                                                    input_dropout=options['transformer']['input_dropout'])
            self.num_layers = options['transformer']['num_layers'] + 1
        self.load_transformer_weights(weight_file)

    def load_transformer_weights(weight_file: str):
        try:
            # TODO(swabha): make more rigorous?
            # device = get_device_of(next(self.parameters()))
            device = -1
            model_state = torch.load(weight_file, map_location=device_mapping(device))

            import ipdb; ipdb.set_trace()
            # TODO(swabha): assign model weights based on all parameter names.
        except Exception:
            raise ConfigurationError("Not Torch File")
