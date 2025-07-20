from decoder import decode_sequences
from utils import test_pairs
import pandas as pd
import random

outputs = []
input = [pair[0] for pair in test_pairs]
ground_truth = [pair[1] for pair in test_pairs]

for i in range(5):
    output_pairs = []
    input_sentence = random.choice(input)
    translated = decode_sequences([input_sentence])
    translated = (
        translated.replace("[PAD]", "")
        .replace("[START]", "")
        .replace("[END]", "")
        .strip()
    )
    output_pairs.append(input_sentence)
    output_pairs.append(translated)
    print(output_pairs)
    outputs.append(output_pairs)

df = pd.DataFrame(outputs, columns=["input sentence", "translation"])
df.to_csv("/Users/adrianajimenez/Desktop/Downloads/REUAICT/Real-Code/2025-ASL-data/seq2seq_code/word_level/babelnet_sampler3.txt", index=False)
