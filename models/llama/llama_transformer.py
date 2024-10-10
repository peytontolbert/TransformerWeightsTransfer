import torch
import torch.nn as nn
from transformers import LlamaPreTrainedModel, Cache, LlamaConfig, LlamaModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding, BaseModelOutputWithPast, DynamicCache, logger
from typing import Optional, Tuple, Union, List

class LLaMATransformer(LlamaPreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    _no_split_modules = ["LlamaDecoderLayer"]
    _tied_weights_keys = ["lm_head.weight", "model.token_embeddings.weight"]

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
        # {{ edit_1 }}
        # Removed redundant redefinition of config
        # config = LlamaConfig(
        #     vocab_size=config.vocab_size,
        #     hidden_size=config.hidden_size,
        #     max_position_embeddings=config.max_position_embeddings,
        #     num_hidden_layers=config.num_hidden_layers,
        #     num_attention_heads=config.num_attention_heads,
        #     intermediate_size=config.intermediate_size,
        #     attention_probs_dropout_prob=config.attention_dropout,
        #     hidden_act=config.hidden_act,
        #     rms_norm_eps=config.rms_norm_eps,
        #     rope_scaling=config.rope_scaling,
        #     rope_theta=config.rope_theta,
        #     torch_dtype=config.torch_dtype,
        #     tie_word_embeddings=config.tie_word_embeddings,
        #     use_cache=config.use_cache
        # )
        # self.config = config
        # {{ edit_1 }}
        print(f"hidden layers: {config.num_hidden_layers}")
        self.padding_idx = config.pad_token_id

        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.positional_encodings = nn.Parameter(torch.zeros(1, config.max_position_embeddings, config.hidden_size))

        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.layer_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        #use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.token_embeddings(input_ids)
        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        causal_mask = self.model._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # Embedding and positional encoding
        #hidden_states = self.token_embeddings(input_ids) + self.positional_encodings[:, :input_ids.size(1), :]

        # Apply LLaMA layers
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                                hidden_states,
                                attention_mask=causal_mask,
                                position_ids=position_ids,
                                past_key_value=past_key_values,
                                output_attentions=output_attentions,
                                use_cache=use_cache,
                                cache_position=cache_position,
                                position_embeddings=position_embeddings,
                            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        # Apply final layer normalization
        hidden_states = self.layer_norm(hidden_states)
        print(f"hidden_states: {hidden_states}")
        # Pass through the language modeling head to get logits
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()
        logits = self.lm_head(hidden_states)
        print(f"logits: {logits}")
        
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attns, logits] if v is not None)
        return hidden_states, logits

    def generate(
        self,
        input_ids,
        max_length=50,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        **generate_kwargs
    ):
        # Generate output using the base model
        outputs = self.model.generate(
            input_ids,
            max_length=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            **generate_kwargs
        )
        # Pass the generated tokens through the lm_head to get logits
        logits = self.lm_head(outputs)
        return outputs, logits
        
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    

    import torch

    def beam_search(model, input_ids, beam_size=3, max_length=20, pad_token_id=0, eos_token_id=2):
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Initialize beam scores (log probabilities)
        beam_scores = torch.zeros((batch_size, beam_size), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -1e9  # Only the first beam is considered at step 0
        beam_seqs = torch.full((batch_size, beam_size, 1), pad_token_id, dtype=torch.long, device=device)

        # Generate until max length or until EOS token is generated
        for step in range(max_length):
            if step == 0:
                # First step, only one sequence per batch
                outputs = model(input_ids)
                logits = outputs[0]  # Assuming logits are returned in the first element
            else:
                # For the next steps, expand beam sequences
                input_ids = beam_seqs.view(batch_size * beam_size, -1)  # Flatten beam_size dimension
                outputs = model(input_ids)
                logits = outputs[0]
            
            # Compute log probabilities
            next_token_logits = logits[:, -1, :]  # Get logits for the last time step
            next_token_log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)

            # Add the current beam scores to the next token probabilities
            next_token_log_probs = next_token_log_probs.view(batch_size, beam_size, -1)
            total_scores = beam_scores.unsqueeze(2) + next_token_log_probs

            # Reshape to collapse the beam dimension and sort the scores
            total_scores = total_scores.view(batch_size, -1)
            top_scores, top_indices = torch.topk(total_scores, beam_size, dim=-1)

            # Update beam sequences and scores
            beam_scores = top_scores
            next_tokens = top_indices % next_token_log_probs.size(-1)
            beam_idx = top_indices // next_token_log_probs.size(-1)

            # Select the best beams
            beam_seqs = torch.cat([beam_seqs[torch.arange(batch_size).unsqueeze(1), beam_idx], next_tokens.unsqueeze(-1)], dim=-1)

            # Check if all beams have generated EOS token
            if (beam_seqs[:, :, -1] == eos_token_id).all():
                break

        # Select the sequence with the highest final score
        best_beams = beam_scores.argmax(dim=-1)
        final_seqs = beam_seqs[torch.arange(batch_size), best_beams]

        return final_seqs

