from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import re
import pickle
import os
import gc
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.simplefilter("ignore")

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# ACTIVER EAGER EXECUTION
tf.config.run_functions_eagerly(True)

default_args = {
    'owner': 'Moustapha Kebe',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Constants
VOCAB_SIZE = 5000
MAX_LENGTH = 50
EMBEDDING_DIM = 64

def load_dataset(**context):
    os.makedirs('/opt/airflow/models', exist_ok=True)
    os.makedirs('/opt/airflow/output', exist_ok=True)
    os.makedirs('/opt/airflow/checkpoints', exist_ok=True)
    
    data = pd.read_csv('/opt/airflow/data/language_detection.csv')
    data.to_csv('/opt/airflow/data/dataset.csv', index=False)
    print(f"Dataset: {len(data)} rows")
    
    n_languages = data['Language'].nunique()
    context['ti'].xcom_push(key='n_languages', value=n_languages)
    
    del data
    gc.collect()
    return True

def encode_labels(**context):
    data = pd.read_csv('/opt/airflow/data/dataset.csv')
    X = data["Text"]
    y = data["Language"]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X.to_csv('/opt/airflow/data/X.csv', index=False, header=['Text'])
    pd.Series(y_encoded).to_csv('/opt/airflow/data/y.csv', index=False, header=['Language'])
    
    with open('/opt/airflow/models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    print(f"Classes: {le.classes_}")
    print(f"Number of classes: {len(le.classes_)}")
    
    context['ti'].xcom_push(key='n_languages', value=len(le.classes_))
    
    del data, X, y, y_encoded, le
    gc.collect()

def clean_text(**context):
    X = pd.read_csv('/opt/airflow/data/X.csv')['Text']
    
    data_list = []
    for text in X:
        text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', text)
        text = re.sub(r'[[]]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        data_list.append(text.strip())
    
    pd.Series(data_list).to_csv('/opt/airflow/data/X_cleaned.csv', index=False, header=['Text'])
    del X, data_list
    gc.collect()

def split_data(**context):
    X = pd.read_csv('/opt/airflow/data/X_cleaned.csv')['Text']
    y = pd.read_csv('/opt/airflow/data/y.csv')['Language']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    X_train.to_csv('/opt/airflow/data/X_train.csv', index=False, header=['Text'])
    X_test.to_csv('/opt/airflow/data/X_test.csv', index=False, header=['Text'])
    pd.Series(y_train).to_csv('/opt/airflow/data/y_train.csv', index=False, header=['Language'])
    pd.Series(y_test).to_csv('/opt/airflow/data/y_test.csv', index=False, header=['Language'])
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    pd.DataFrame({'text': X_test, 'label': y_test}).to_csv('/opt/airflow/data/test_data.csv', index=False)
    
    del X, y, X_train, X_test, y_train, y_test
    gc.collect()

def prepare_dl_data(**context):
    """Prepare data for deep learning model"""
    X_train = pd.read_csv('/opt/airflow/data/X_train.csv')['Text']
    X_test = pd.read_csv('/opt/airflow/data/X_test.csv')['Text']
    y_train = pd.read_csv('/opt/airflow/data/y_train.csv')['Language']
    y_test = pd.read_csv('/opt/airflow/data/y_test.csv')['Language']
    
    n_languages = context['ti'].xcom_pull(task_ids='encode_labels', key='n_languages')
    
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Limiter les indices
    X_train_seq = [[min(idx, VOCAB_SIZE-1) for idx in seq] for seq in X_train_seq]
    X_test_seq = [[min(idx, VOCAB_SIZE-1) for idx in seq] for seq in X_test_seq]
    
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
    
    y_train_cat = to_categorical(y_train, num_classes=n_languages)
    y_test_cat = to_categorical(y_test, num_classes=n_languages)
    
    np.save('/opt/airflow/data/X_train_pad.npy', X_train_pad)
    np.save('/opt/airflow/data/X_test_pad.npy', X_test_pad)
    np.save('/opt/airflow/data/y_train_cat.npy', y_train_cat)
    np.save('/opt/airflow/data/y_test_cat.npy', y_test_cat)
    
    with open('/opt/airflow/models/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    print(f"Vocabulary size: {len(tokenizer.word_index)}")
    print(f"Training data shape: {X_train_pad.shape}")
    
    del X_train, X_test, y_train, y_test, X_train_seq, X_test_seq, X_train_pad, X_test_pad, y_train_cat, y_test_cat, tokenizer
    gc.collect()

def build_dl_model(**context):
    """Build deep learning model"""
    n_languages = context['ti'].xcom_pull(task_ids='encode_labels', key='n_languages')
    
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LENGTH),
        Conv1D(64, 3, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'), 
        Dropout(0.2),
        Dense(n_languages, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    model_json = model.to_json()
    with open('/opt/airflow/models/dl_model_architecture.json', 'w') as f:
        f.write(model_json)
    
    context['ti'].xcom_push(key='model_architecture', value=model_json)

def train_dl_model(**context):
    """Train the deep learning model"""
    X_train = np.load('/opt/airflow/data/X_train_pad.npy')
    y_train = np.load('/opt/airflow/data/y_train_cat.npy')
    X_test = np.load('/opt/airflow/data/X_test_pad.npy')
    y_test = np.load('/opt/airflow/data/y_test_cat.npy')
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    X_train = np.clip(X_train, 0, VOCAB_SIZE-1)
    X_test = np.clip(X_test, 0, VOCAB_SIZE-1)
    
    with open('/opt/airflow/models/dl_model_architecture.json', 'r') as f:
        model_architecture = f.read()
    
    model = tf.keras.models.model_from_json(model_architecture)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    ]
    
    print("🚀 Starting training...")
    
    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    model.save('/opt/airflow/models/dl_model.h5')
    
    with open('/opt/airflow/models/training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    
    print("✅ Training completed!")
    
    final_accuracy = history.history.get('val_accuracy', history.history.get('accuracy'))[-1]
    print(f"Final accuracy: {final_accuracy:.4f}")
    context['ti'].xcom_push(key='final_accuracy', value=final_accuracy)
    
    del model, X_train, y_train, X_test, y_test, history
    gc.collect()

def evaluate_dl_model(**context):
    """Evaluate deep learning model"""
    model = tf.keras.models.load_model('/opt/airflow/models/dl_model.h5')
    X_test = np.load('/opt/airflow/data/X_test_pad.npy')
    y_test_cat = np.load('/opt/airflow/data/y_test_cat.npy')
    
    X_test = np.clip(X_test, 0, VOCAB_SIZE-1)
    
    with open('/opt/airflow/models/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    
    test_data = pd.read_csv('/opt/airflow/data/test_data.csv')
    
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test_cat, axis=1)
    
    accuracy = accuracy_score(y_true, y_pred)
    
    results_df = pd.DataFrame({
        'text': test_data['text'],
        'actual': [le.classes_[i] for i in y_true],
        'predicted': [le.classes_[i] for i in y_pred],
        'confidence': np.max(y_pred_proba, axis=1),
        'correct': y_true == y_pred
    })
    
    results_df.to_csv('/opt/airflow/output/dl_predictions.csv', index=False)
    
    evaluation_metrics = {
        'accuracy': float(accuracy),
        'model_type': 'deep_learning',
        'timestamp': datetime.now().isoformat(),
        'vocab_size': VOCAB_SIZE,
        'max_length': MAX_LENGTH
    }
    
    with open('/opt/airflow/output/dl_evaluation_metrics.json', 'w') as f:
        json.dump(evaluation_metrics, f, indent=2)
    
    print(f"✅ Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    
    context['ti'].xcom_push(key='dl_accuracy', value=float(accuracy))
    
    del model, X_test, y_test_cat, y_pred_proba
    gc.collect()


with DAG(
    'deep_learning_language_detection',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['dl', 'nlp', 'deep-learning'],
) as dag:
    
    t1 = PythonOperator(task_id='load_dataset', python_callable=load_dataset)
    t2 = PythonOperator(task_id='encode_labels', python_callable=encode_labels)
    t3 = PythonOperator(task_id='clean_text', python_callable=clean_text)
    t4 = PythonOperator(task_id='split_data', python_callable=split_data)
    t5 = PythonOperator(task_id='prepare_dl_data', python_callable=prepare_dl_data)
    t6 = PythonOperator(task_id='build_dl_model', python_callable=build_dl_model)
    t7 = PythonOperator(task_id='train_dl_model', python_callable=train_dl_model)
    t8 = PythonOperator(task_id='evaluate_dl_model', python_callable=evaluate_dl_model)
    
    t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7 >> t8