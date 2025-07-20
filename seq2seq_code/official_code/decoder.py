from keras import ops
from utils import max_length
from utils import eng_tokenizer, asl_tokenizer
from sample import VocabSampler, BabelNetSampler
from train import transformer
from utils import test_pairs
import random
import pandas as pd
import tensorflow as tf


MAX_SEQUENCE_LENGTH = max_length

def decode_sequences(input_sentences):
    with tf.device('/CPU:0'):
        batch_size = 1
        prompt_length = MAX_SEQUENCE_LENGTH   # still pad encoder/prompt to fixed size

        # — Encoder input preparation (unchanged) —
        encoder_inputs = ops.convert_to_tensor(eng_tokenizer(input_sentences))
        seq_len = tf.shape(encoder_inputs)[1]
        if seq_len < prompt_length:
            pad_amt = prompt_length - seq_len
            pads    = ops.full((1, pad_amt), 0, dtype=encoder_inputs.dtype)
            encoder_inputs = ops.concatenate([encoder_inputs, pads], axis=1)

        # — Compute dynamic decode lengths —
        input_ids      = eng_tokenizer(input_sentences)[0]
        input_len      = len(input_ids)
        max_decode_len = min(prompt_length, input_len)               # at most input+2
        min_decode_len = max(1, int(input_len * 0.75))                   # at least 80% of input

        # — Initialize your sampler as before —
        sampler = BabelNetSampler(
            sentence=input_sentences[0].split(),
            vocab=asl_tokenizer.token_to_id_map,
        )

        # — Build initial prompt ([START] + PADs) —
        start_id = asl_tokenizer.token_to_id("[START]")
        pad_id   = asl_tokenizer.token_to_id("[PAD]")
        prompt   = ops.full((batch_size, prompt_length), pad_id, dtype=tf.int32)
        start_t  = ops.convert_to_tensor([[start_id]], dtype=tf.int32)
        prompt   = ops.slice_update(prompt, (0, 0), start_t)

        # — Next‑token fn (unchanged) —
        def next_fn(pr, cache, idx):
            logits = transformer([encoder_inputs, pr])[:, idx - 1, :]
            return logits, None, cache

        cache     = None
        generated = []

        # — Decoding loop with dynamic bounds —
        for idx in range(1, max_decode_len):
            logits, _, cache = next_fn(prompt, cache, idx)
            token = sampler.get_next_token(logits)                # shape [1]
            token = ops.cast(token, dtype=prompt.dtype)
            prompt = ops.slice_update(prompt, (0, idx), ops.expand_dims(token, 0))

            tok_id = int(token.numpy()[0])
            generated.append(tok_id)

            # only stop if we’ve hit [END] *and* run at least min_decode_len steps
            if tok_id == asl_tokenizer.token_to_id("[END]") and idx >= min_decode_len:
                break

        return asl_tokenizer.detokenize(generated)