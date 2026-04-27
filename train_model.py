# next_word_prediction.py

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense, Dropout

# -------------------------------
# DATASET
# -------------------------------
corpus = [
    "machine learning is a subset of artificial intelligence",
    "neural networks are inspired by the human brain",
    "data science involves statistics and programming",
    "deep learning uses multiple layers of neurons",
    "python is widely used for data analysis",
    "artificial intelligence is transforming industries",
    "algorithms are used to solve problems efficiently",
    "big data refers to large complex datasets",
    "supervised learning uses labeled data",
    "unsupervised learning finds hidden patterns",

    "machine learning models learn from data",
    "deep learning models require large datasets",
    "neural networks consist of layers of nodes",
    "data analysis helps in decision making",
    "python supports multiple programming paradigms",
    "artificial intelligence mimics human intelligence",
    "statistics is important for data science",
    "learning algorithms improve with experience",
    "models are evaluated using test data",
    "training data is used to train models",

    "classification is a supervised learning task",
    "regression predicts continuous values",
    "clustering groups similar data points",
    "feature engineering improves model performance",
    "data preprocessing is a crucial step",
    "overfitting occurs when model memorizes data",
    "underfitting occurs when model is too simple",
    "validation data helps tune hyperparameters",
    "optimization algorithms minimize loss function",
    "activation functions introduce nonlinearity"
]

# -------------------------------
# PREPROCESSING
# -------------------------------
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

total_words = len(tokenizer.word_index) + 1

input_sequences = []

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i+1])

max_len = max(len(seq) for seq in input_sequences)

input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# -------------------------------
# RNN MODEL
# -------------------------------
def build_rnn():
    model = Sequential([
        Embedding(total_words, 64),
        SimpleRNN(128),
        Dense(total_words, activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


# -------------------------------
# LSTM MODEL
# -------------------------------
def build_lstm():
    model = Sequential([
        Embedding(total_words, 64),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(128),
        Dense(128, activation='relu'),
        Dense(total_words, activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


# -------------------------------
# TRAIN MODELS
# -------------------------------
print("Training RNN model...")
rnn_model = build_rnn()
rnn_model.fit(X, y, epochs=50, verbose=1)

print("\nTraining LSTM model...")
lstm_model = build_lstm()
lstm_model.fit(X, y, epochs=50, verbose=1)

# -------------------------------
# SAVE MODELS & TOKENIZER
# -------------------------------
import pickle

# Save trained models
rnn_model.save("rnn_model.h5")
lstm_model.save("lstm_model.h5")

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Save config (important for app.py)
with open("config.pkl", "wb") as f:
    pickle.dump({"max_len": max_len}, f)

print("\n✅ Models, tokenizer, and config saved successfully!")


# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict_next_word(model, text, tokenizer, max_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')

    predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)

    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word

    return ""


# -------------------------------
# TESTING
# -------------------------------
print("\n--- TESTING ---")
seed_text = "machine learning is"

print("Input:", seed_text)
print("RNN Prediction:", predict_next_word(rnn_model, seed_text, tokenizer, max_len))
print("LSTM Prediction:", predict_next_word(lstm_model, seed_text, tokenizer, max_len))
