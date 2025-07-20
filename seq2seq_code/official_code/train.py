from utils import max_length
from utils import eng_tokenizer, asl_tokenizer
from utils import train_pairs, val_pairs
from utils import num_encoder_tokens, num_decoder_tokens

import keras
import keras_hub

import tensorflow as tf
import tensorflow.data as tf_data
from tensorflow_text.tools.wordpiece_vocab import (
    bert_vocab_from_dataset,
)

MAX_SEQUENCE_LENGTH = max_length
BATCH_SIZE = 16
EPOCHS = 20
EMBED_DIM = 64
INTERMEDIATE_DIM = 128
NUM_HEADS = 4

def preprocess_batch(eng, asl):
    eng_start_end_packer = keras_hub.layers.StartEndPacker(
        sequence_length=MAX_SEQUENCE_LENGTH,
        pad_value=eng_tokenizer.token_to_id("[PAD]"),
        dtype="int32"
    )
    eng = eng_start_end_packer(eng)

    asl_start_end_packer = keras_hub.layers.StartEndPacker(
        sequence_length=MAX_SEQUENCE_LENGTH + 1,
        start_value=asl_tokenizer.token_to_id("[START]"),
        end_value=asl_tokenizer.token_to_id("[END]"),
        pad_value=asl_tokenizer.token_to_id("[PAD]"),
        dtype="int32"
    )
    asl = asl_start_end_packer(asl)

    decoder_inputs = asl[:, :-1]
    decoder_outputs = asl[:, 1:]

    return {
        "encoder_inputs": eng,
        "decoder_inputs": decoder_inputs
    }, decoder_outputs


def make_dataset(pairs):
    
    eng_ids = [eng_tokenizer.tokenize(sent) for sent, _ in pairs]    
    asl_ids = [asl_tokenizer.tokenize(sent) for _, sent in pairs]

    # 🛠️ Force token type to int32
    eng_tensor = tf.ragged.constant(eng_ids, dtype=tf.int32)
    asl_tensor = tf.ragged.constant(asl_ids, dtype=tf.int32)
    
    dataset = tf_data.Dataset.from_tensor_slices((eng_tensor, asl_tensor))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(preprocess_batch, num_parallel_calls=tf_data.AUTOTUNE)
    return dataset.shuffle(2048).prefetch(16).cache()

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)
print(train_ds)


# Encoder
encoder_inputs = keras.Input(shape=(None,), name="encoder_inputs")

x = keras_hub.layers.TokenAndPositionEmbedding(
    vocabulary_size=num_encoder_tokens,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
)(encoder_inputs)

encoder_outputs = keras_hub.layers.TransformerEncoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(inputs=x)
encoder = keras.Model(encoder_inputs, encoder_outputs)


# Decoder
decoder_inputs = keras.Input(shape=(None,), name="decoder_inputs")
encoded_seq_inputs = keras.Input(shape=(None, EMBED_DIM), name="decoder_state_inputs")

x = keras_hub.layers.TokenAndPositionEmbedding(
    vocabulary_size=num_decoder_tokens,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
)(decoder_inputs)

x = keras_hub.layers.TransformerDecoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(decoder_sequence=x, encoder_sequence=encoded_seq_inputs)
x = keras.layers.Dropout(0.5)(x)
decoder_outputs = keras.layers.Dense(num_decoder_tokens, activation="softmax")(x)
decoder = keras.Model(
    [
        decoder_inputs,
        encoded_seq_inputs,
    ],
    decoder_outputs,
)
decoder_outputs = decoder([decoder_inputs, encoder_outputs])

transformer = keras.Model(
    [encoder_inputs, decoder_inputs],
    decoder_outputs,
    name="transformer",
)

optimizer = keras.optimizers.Adam(learning_rate=1e-4)

transformer.summary()
transformer.compile(
    optimizer, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"]
)
transformer.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)