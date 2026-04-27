# 🧠 Next Word Prediction App

A simple NLP application that predicts the **next word in a sequence** using **RNN** and **LSTM** models. Built with **TensorFlow** and deployed using **Streamlit**.

---

## 🚀 Features

* Predict next word from input text
* Choose between **RNN** and **LSTM**
* Interactive Streamlit UI
* Custom-trained NLP model

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Streamlit

---

## 📂 Project Files

```
app.py              # Streamlit app
train_model.py      # Training script
rnn_model.h5        # Trained RNN model
lstm_model.h5       # Trained LSTM model
requirements.txt
README.md
```

---

## ⚙️ How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Train the Model (Run First)

```bash
python train_model.py
```

This will create:

* `rnn_model.h5`
* `lstm_model.h5`

---

### 3️⃣ Run the App

```bash
streamlit run app.py
```

---


---

## 📌 Notes

* Always run `train_model.py` before `app.py`

---

