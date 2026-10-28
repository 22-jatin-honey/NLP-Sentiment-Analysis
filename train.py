"""Train a Bidirectional LSTM sentiment classifier on the Keras IMDB dataset."""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

SEED = 42
MAX_FEATURES = 30000
MAX_LEN = 250
BATCH_SIZE = 128
EPOCHS = 3
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

tf.keras.utils.set_random_seed(SEED)
print("Loading IMDB dataset (first run may download it)...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_FEATURES, seed=SEED)
x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding="pre", truncating="pre")
x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding="pre", truncating="pre")

model = Sequential([
    Embedding(input_dim=MAX_FEATURES, output_dim=128, input_length=MAX_LEN, mask_zero=True),
    Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.0)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")])
model.summary()
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(str(ARTIFACTS / "bilstm_sentiment.keras"), monitor="val_loss", save_best_only=True),
]
model.fit(x_train, np.asarray(y_train), validation_split=0.15, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1)

best_model = tf.keras.models.load_model(ARTIFACTS / "bilstm_sentiment.keras")
results = best_model.evaluate(x_test, np.asarray(y_test), batch_size=BATCH_SIZE, verbose=1, return_dict=True)
print("\nTest metrics:")
for name, value in results.items():
    print(f"{name}: {value:.4f}")
with open(ARTIFACTS / "test_metrics.json", "w", encoding="utf-8") as f:
    json.dump({k: float(v) for k, v in results.items()}, f, indent=2)

# The Keras IMDB vocabulary uses indices beginning at 1. The model input adds 3
# to each known word index (0/1/2 are reserved for padding/start/unknown).
word_index = imdb.get_word_index()
with open(ARTIFACTS / "word_index.json", "w", encoding="utf-8") as f:
    json.dump(word_index, f)
print(f"\nSaved model and vocabulary in: {ARTIFACTS}")
