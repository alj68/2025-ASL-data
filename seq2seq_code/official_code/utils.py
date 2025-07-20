import pyparsing as pp
import re
import random
import tensorflow as tf

from grammar import full_grammar as fg

# generate
    # 1) list of eng-asl sentence pairs
    # 2) set of unique english vocab
    # 3) set of unique asl vocab
data_path = "/Users/adrianajimenez/Desktop/Downloads/REUAICT/Real-Code/2025-ASL-data/sent_pairs_joined.txt"
text_pairs = []
eng_texts = []
asl_texts = []
SPECIAL_TOKENS = ["[PAD]", "[START]", "[END]", "[UNK]"]
eng_tokens = set(SPECIAL_TOKENS)
asl_tokens = set(SPECIAL_TOKENS)
max_length = 0

class DictTokenizer:
    def __init__(self, vocab, tokenizer_fn):
        self.token_to_id_map = vocab
        self.id_to_token_map = {i: t for t, i in vocab.items()}
        self.tokenizer_fn = tokenizer_fn

    def __call__(self, text_batch):
        return [
            [self.token_to_id_map.get(tok, self.token_to_id_map.get("[UNK]", 0)) 
             for tok in self.tokenizer_fn(text)]
            for text in text_batch
        ]

    def tokenize(self, text):
        return [self.token_to_id_map.get(tok, self.token_to_id_map.get("[UNK]", 0)) 
                for tok in self.tokenizer_fn(text)]

    def detokenize(self, token_ids):
        if isinstance(token_ids, tf.Tensor):
            token_ids = token_ids.numpy()
        elif isinstance(token_ids, tf.RaggedTensor):
            token_ids = token_ids.to_tensor().numpy()
        elif isinstance(token_ids, int):
            token_ids = [token_ids]

        return " ".join([self.id_to_token_map.get(int(tok_id), "[UNK]") for tok_id in token_ids])

    def token_to_id(self, token):
        return self.token_to_id_map.get(token, self.token_to_id_map.get("[UNK]", 0))
    
# tokenize based on predefined grammar rules

def custom_asl_tokenize(text):
    try:
        if "'" in text:
            text = text.replace("'", "")
        if "++" in text:
            text = text.replace("++", "+")
        return fg.parse_string(text, parse_all=True).asList()
    except pp.ParseException as pe:
        print(text)
        print(f"Failed to parse: {pe}")
        return []
    
def custom_eng_tokenize(text):
    # Perserve punctuation and digits
    text = re.sub(r'([^\w\s]|\d)', r' \1 ', text)
    # Convert to lowercase
    text = text.lower()
    # Split on whitespace
    tokens = text.split()
    return tokens

with open(data_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")

for line in lines:
    pair = []
    eng_text, asl_text = line.split("\t")
    eng_texts.append(eng_text)
    asl_texts.append(asl_text)
    pair.append(eng_text.lower())
    pair.append(asl_text)
    text_pairs.append(pair)
    
for text in eng_texts:
    tokens = custom_eng_tokenize(text)
    length = len(tokens)
    if length > max_length:
        max_length = length
    for token in tokens:
        if token not in eng_tokens:
                eng_tokens.add(token)
            
for text in asl_texts:
    tokens = custom_asl_tokenize(text)
    length = len(tokens)
    if length > max_length:
        max_length = length
    for token in tokens:
        if token not in asl_tokens:
                asl_tokens.add(token)
                            
max_encoder_seq_length = max([len(txt) for txt in eng_texts])
max_decoder_seq_length = max([len(txt) for txt in asl_texts])

eng_tokens = sorted(list(eng_tokens))
asl_tokens = sorted(list(asl_tokens))

print("eng_tokens:", eng_tokens)
print("asl_tokens", asl_tokens)
num_encoder_tokens = len(eng_tokens)
num_decoder_tokens = len(asl_tokens)
print("num_eng_tokens", num_encoder_tokens)
print("num_asl_tokens", num_decoder_tokens)

# split data

random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]

print(f"{len(text_pairs)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")

eng_vocab = dict([(char, i) for i, char in enumerate(eng_tokens)])
asl_vocab = dict([(char, i) for i, char in enumerate(asl_tokens)])

eng_tokenizer = DictTokenizer(eng_vocab, tokenizer_fn=custom_eng_tokenize)
asl_tokenizer = DictTokenizer(asl_vocab, tokenizer_fn=custom_asl_tokenize)

print(eng_tokenizer)
print(asl_tokenizer)

print(eng_vocab)
print(asl_vocab)



    