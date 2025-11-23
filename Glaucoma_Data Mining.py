#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM,
    Conv1D, GlobalMaxPooling1D, Dense, Dropout,
    Concatenate
)
from tensorflow.keras.models import Model


# ===================== CONFIG =====================


DATA_PATH = r"C:\Users\kamol\Desktop\Python\glaucoma.csv"

TEXT_COL  = "note"
LABEL_COL = "glaucoma"
RACE_COL  = "race"

MAX_NUM_WORDS = 20000
MAX_LEN       = 256
EMBED_DIM     = 128
BATCH_SIZE    = 32
EPOCHS        = 5
LR            = 1e-3


# ===================== DATA LOADING =====================

def load_data(path):
    df = pd.read_csv(path)

    df = df.dropna(subset=[TEXT_COL, LABEL_COL])

    y = (df[LABEL_COL].astype(str).str.lower() == "yes").astype(int).values
    texts = df[TEXT_COL].astype(str).values
    races = df[RACE_COL].astype(str).str.capitalize().values

    drop_cols = [TEXT_COL, LABEL_COL, RACE_COL]
    if "use" in df.columns:
        drop_cols.append("use")

    demo_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return texts, y, races, demo_df


# ===================== PREPROCESSING =====================

def preprocess(texts, y, races, demo_df):
    X_text_train, X_text_test, y_train, y_test, races_train, races_test, demo_train, demo_test =         train_test_split(
            texts, y, races, demo_df,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<UNK>")
    tokenizer.fit_on_texts(X_text_train)

    seq_train = tokenizer.texts_to_sequences(X_text_train)
    seq_test  = tokenizer.texts_to_sequences(X_text_test)

    X_train_seq = pad_sequences(seq_train, maxlen=MAX_LEN, padding='post', truncating='post')
    X_test_seq  = pad_sequences(seq_test,  maxlen=MAX_LEN, padding='post', truncating='post')

    vocab_size = min(MAX_NUM_WORDS, len(tokenizer.word_index) + 1)

    num_cols = demo_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = demo_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )

    X_train_demo = preprocessor.fit_transform(demo_train)
    X_test_demo  = preprocessor.transform(demo_test)

    demographic_dim = X_train_demo.shape[1]

    return (X_train_seq, X_test_seq,
            X_train_demo, X_test_demo,
            y_train, y_test,
            races_train, races_test,
            vocab_size, demographic_dim)


# ===================== MODELS =====================

def build_lstm_model(vocab_size, embed_dim, max_len, demographic_dim):

    text_input = Input(shape=(max_len,), name="text_input")
    x = Embedding(vocab_size, embed_dim, input_length=max_len)(text_input)
    x = Bidirectional(LSTM(128))(x)

    demo_input = Input(shape=(demographic_dim,), name="demo_input")
    demo_dense = Dense(64, activation='relu')(demo_input)

    fused = Concatenate()([x, demo_dense])
    fused = Dense(128, activation='relu')(fused)
    fused = Dropout(0.5)(fused)

    output = Dense(1, activation='sigmoid')(fused)

    model = Model(inputs=[text_input, demo_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_cnn_model(vocab_size, embed_dim, max_len, demographic_dim):

    text_input = Input(shape=(max_len,), name="text_input")
    x = Embedding(vocab_size, embed_dim, input_length=max_len)(text_input)

    conv3 = Conv1D(128, kernel_size=3, activation='relu')(x)
    conv4 = Conv1D(128, kernel_size=4, activation='relu')(x)
    conv5 = Conv1D(128, kernel_size=5, activation='relu')(x)

    pool3 = GlobalMaxPooling1D()(conv3)
    pool4 = GlobalMaxPooling1D()(conv4)
    pool5 = GlobalMaxPooling1D()(conv5)

    cnn_features = Concatenate()([pool3, pool4, pool5])

    demo_input = Input(shape=(demographic_dim,), name="demo_input")
    demo_dense = Dense(64, activation='relu')(demo_input)

    fused = Concatenate()([cnn_features, demo_dense])
    fused = Dense(128, activation='relu')(fused)
    fused = Dropout(0.5)(fused)

    output = Dense(1, activation='sigmoid')(fused)

    model = Model(inputs=[text_input, demo_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ===================== METRICS =====================

def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = (tp + tn) / len(y_true)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    return accuracy, auc, sensitivity, specificity


def group_auc(y_true, y_prob, races):
    groups = ["Asian", "Black", "White"]
    result = {}
    for g in groups:
        mask = (races == g)
        if mask.sum() > 1:
            result[g] = roc_auc_score(y_true[mask], y_prob[mask])
        else:
            result[g] = np.nan
    return result


def print_results(name, y_test, y_prob, races_test):
    acc, auc, sens, spec = compute_metrics(y_test, y_prob)

    print(f"\n========== {name} MODEL ==========")
    print("Accuracy          :", round(acc,4))
    print("AUC               :", round(auc,4))
    print("Sensitivity (TPR) :", round(sens,4))
    print("Specificity (TNR) :", round(spec,4))

    group_res = group_auc(y_test, y_prob, races_test)
    print("\n=== Group-wise AUCs ===")
    for g, v in group_res.items():
        print(f"{g} AUC :", v)


# ===================== MAIN =====================

def main():
    print("Loading dataset from:", DATA_PATH)
    texts, y, races, demo_df = load_data(DATA_PATH)

    (X_train_seq, X_test_seq,
     X_train_demo, X_test_demo,
     y_train, y_test,
     races_train, races_test,
     vocab_size, demographic_dim) = preprocess(texts, y, races, demo_df)

    if not isinstance(X_train_demo, np.ndarray):
        X_train_demo = X_train_demo.toarray()
        X_test_demo  = X_test_demo.toarray()

    print("\nTraining LSTM model...")
    lstm_model = build_lstm_model(vocab_size, EMBED_DIM, MAX_LEN, demographic_dim)
    lstm_model.fit(
        [X_train_seq, X_train_demo],
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )
    lstm_prob = lstm_model.predict([X_test_seq, X_test_demo]).ravel()
    print_results("LSTM", y_test, lstm_prob, races_test)


    print("\nTraining CNN model...")
    cnn_model = build_cnn_model(vocab_size, EMBED_DIM, MAX_LEN, demographic_dim)
    cnn_model.fit(
        [X_train_seq, X_train_demo],
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )
    cnn_prob = cnn_model.predict([X_test_seq, X_test_demo]).ravel()
    print_results("CNN", y_test, cnn_prob, races_test)


if __name__ == "__main__":
    main()


# In[ ]:




