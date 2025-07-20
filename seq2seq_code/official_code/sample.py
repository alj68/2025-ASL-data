import keras
import keras_hub
import tensorflow as tf
import requests

class VocabSampler(keras_hub.samplers.Sampler):
    def __init__(self, sentence, vocab,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_words    = [w for w in sentence if len(w) > 2]
        self.vocab          = vocab
        self.prefered_boost = 2.0
        self.allowed_boost  = 1.5
        self.temperature    = 0.8
        self.top_k          = 10               # store it
        self.max_repeats    = 2
        # precompute once
        self.allowed_ids    = self.compute_allowed_ids("allowed")
        self.prefered_ids   = self.compute_allowed_ids("prefer")
        self.prev_token     = None
        self.repeat_count   = 0

    def compute_allowed_ids(self, preference):
        prefer, allowed = set(), set()
        for word in self.input_words:
            p, s = word[:3], word[-3:]
            for gloss, gid in self.vocab.items():
                if gloss.startswith(p) and gloss.endswith(s):
                    prefer.add(gid)
                elif gloss.startswith(p) or gloss.endswith(s):
                    allowed.add(gid)
        return prefer if preference=="prefer" else allowed

    def get_next_token(self, probabilities):
        # 1) Boost matching gloss logits
        for gid in self.allowed_ids:
            boosted = probabilities[0, gid] + self.allowed_boost
            probabilities  = keras.ops.slice_update(
                probabilities, (0, gid),
                keras.ops.reshape(boosted, (1,1))
            )
        for gid in self.prefered_ids:
            boosted = probabilities[0, gid] + self.prefered_boost
            probabilities  = keras.ops.slice_update(
                probabilities, (0, gid),
                keras.ops.reshape(boosted, (1,1))
            )

        # 2) Repetition penalty
        if self.prev_token is not None and self.repeat_count >= self.max_repeats:
            neg_inf = tf.constant([[-1e9]], dtype=probabilities.dtype)
            logits  = keras.ops.slice_update(probabilities, (0, self.prev_token), neg_inf)

        # 3) Temperature scaling
        scaled_logits = probabilities / tf.cast(self.temperature, probabilities.dtype)

        # 4) Top‑k filtering via threshold
        topk_vals, _ = tf.math.top_k(scaled_logits, k=self.top_k)
        threshold    = tf.reduce_min(topk_vals, axis=-1, keepdims=True)  # [batch,1]
        neg_inf      = tf.constant(-1e9, dtype=scaled_logits.dtype)
        filtered     = tf.where(
            scaled_logits < threshold,
            neg_inf,
            scaled_logits
        )

        # 5) Sample from the filtered logits
        # tf.random.categorical expects 2D [batch, vocab_size]
        sample_id = tf.random.categorical(filtered, num_samples=1)  # [batch,1]
        token     = tf.squeeze(sample_id, axis=1)                  # [batch]

        # 6) Update repeat tracking
        tid = int(token.numpy()[0])
        if tid == self.prev_token:
            self.repeat_count += 1
        else:
            self.prev_token   = tid
            self.repeat_count = 1

        return token  # shape [1]


class BabelNetSampler(keras_hub.samplers.Sampler):
    def __init__(self, sentence, vocab,
                 key = "70426f32-2291-4d4e-9f6d-a220bc1a93f6",
                 lang = "EN",
                 target_lan = "EN",
                 id_url = "https://babelnet.io/v9/getSynsetIds",
                 word_url = "https://babelnet.io/v9/getSenses",
                 **kwargs):
        super().__init__(**kwargs)
        self.input_words    = [w for w in sentence if len(w) > 2]
        self.vocab          = vocab
        self.prefered_boost = 2.0
        self.allowed_boost  = 1.5
        self.temperature    = 0.8
        self.top_k          = 10               # store it
        self.max_repeats    = 2
        self.id_url = id_url
        self.word_url = word_url
        self.key = key
        self.target = target_lan
        self.lang = lang
        # precompute once
        self.allowed_ids    = self.compute_allowed_ids("allowed")
        self.prefered_ids   = self.compute_allowed_ids("prefer")
        self.prev_token     = None
        self.repeat_count   = 0
    
    def fetch_senses(self, word):
        word = word.lower()
        header = {'Accept-Encoding':'gzip'}
        word_params = {"lemma": word, "searchLang": self.lang, "targetLang":self.target, "key": self.key}
        candidates = set()
        try:
            resp = requests.get(self.word_url, params=word_params, headers=header)
            resp.raise_for_status()
            data = resp.json()
            for each in data:
                if each.get("type") == "BabelSense":
                    props = each.get("properties", {})
                    lemmas = props.get("lemma", {})
                    lemma = lemmas.get("lemma")                    
                    if lemma.isalpha() and (lemma.upper() not in candidates):
                        candidates.add(lemma.upper())
        except Exception as e:
            print(f"BabelNet fetch error for '{word}':", e)
            raise
        
        return candidates
            

    def compute_allowed_ids(self, preference):
        prefer, allowed = set(), set()
        for word in self.input_words:
            candidates = self.fetch_senses(word)
            print(candidates)
            for each in candidates:
                p, s = each[:3], each[-3:]
                print(p)
                print(s)
                for gloss, gid in self.vocab.items():
                    if gloss.startswith(p) and gloss.endswith(s):
                        prefer.add(gid)
                    elif gloss.startswith(p) or gloss.endswith(s):
                        allowed.add(gid)
        print(prefer)
        print(allowed)
        return prefer if preference=="prefer" else allowed

    def get_next_token(self, probabilities):
        # 1) Boost matching gloss logits
        for gid in self.allowed_ids:
            print(gid)
            boosted = probabilities[0, gid] + self.allowed_boost
            probabilities  = keras.ops.slice_update(
                probabilities, (0, gid),
                keras.ops.reshape(boosted, (1,1))
            )
        for gid in self.prefered_ids:
            print(gid)
            boosted = probabilities[0, gid] + self.prefered_boost
            logits  = keras.ops.slice_update(
                probabilities, (0, gid),
                keras.ops.reshape(boosted, (1,1))
            )

        # 2) Repetition penalty
        if self.prev_token is not None and self.repeat_count >= self.max_repeats:
            neg_inf = tf.constant([[-1e9]], dtype=probabilities.dtype)
            logits  = keras.ops.slice_update(probabilities, (0, self.prev_token), neg_inf)

        # 3) Temperature scaling
        scaled_logits = probabilities / tf.cast(self.temperature, probabilities.dtype)

        # 4) Top‑k filtering via threshold
        topk_vals, _ = tf.math.top_k(scaled_logits, k=self.top_k)
        threshold    = tf.reduce_min(topk_vals, axis=-1, keepdims=True)  # [batch,1]
        neg_inf      = tf.constant(-1e9, dtype=scaled_logits.dtype)
        filtered     = tf.where(
            scaled_logits < threshold,
            neg_inf,
            scaled_logits
        )

        # 5) Sample from the filtered logits
        # tf.random.categorical expects 2D [batch, vocab_size]
        sample_id = tf.random.categorical(filtered, num_samples=1)  # [batch,1]
        token     = tf.squeeze(sample_id, axis=1)                  # [batch]

        # 6) Update repeat tracking
        tid = int(token.numpy()[0])
        if tid == self.prev_token:
            self.repeat_count += 1
        else:
            self.prev_token   = tid
            self.repeat_count = 1

        return token  # shape [1]