from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import re
import pickle
import os
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import warnings
warnings.simplefilter("ignore")

default_args = {
    'owner': 'Moustapha Kebe',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def load_dataset(**context):
    os.makedirs('/opt/airflow/models', exist_ok=True)
    os.makedirs('/opt/airflow/output', exist_ok=True)
    
    data = pd.read_csv('/opt/airflow/data/language_detection.csv')
    data.to_csv('/opt/airflow/data/dataset.csv', index=False)
    print(f"Dataset: {len(data)} rows")
    del data
    gc.collect()
    return True

def encode_labels(**context):
    data = pd.read_csv('/opt/airflow/data/dataset.csv')
    X = data["Text"]
    y = data["Language"]
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    X.to_csv('/opt/airflow/data/X.csv', index=False, header=['Text'])
    pd.Series(y).to_csv('/opt/airflow/data/y.csv', index=False, header=['Language'])
    
    with open('/opt/airflow/models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    print(f"Classes: {le.classes_}")
    del data, X, y, le
    gc.collect()

def clean_text(**context):
    X = pd.read_csv('/opt/airflow/data/X.csv')['Text']
    
    data_list = []
    for text in X:
        text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', text)
        text = re.sub(r'[[]]', ' ', text)
        text = text.lower()
        data_list.append(text)
    
    pd.Series(data_list).to_csv('/opt/airflow/data/X_cleaned.csv', index=False, header=['Text'])
    del X, data_list
    gc.collect()

def split_data(**context):
    X = pd.read_csv('/opt/airflow/data/X.csv')['Text']
    y = pd.read_csv('/opt/airflow/data/y.csv')['Language']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    X_train.to_csv('/opt/airflow/data/X_train.csv', index=False, header=['Text'])
    X_test.to_csv('/opt/airflow/data/X_test.csv', index=False, header=['Text'])
    pd.Series(y_train).to_csv('/opt/airflow/data/y_train.csv', index=False, header=['Language'])
    pd.Series(y_test).to_csv('/opt/airflow/data/y_test.csv', index=False, header=['Language'])
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    del X, y, X_train, X_test, y_train, y_test
    gc.collect()

def create_bag_of_words(**context):
    X_train = pd.read_csv('/opt/airflow/data/X_train.csv')['Text']
    X_test = pd.read_csv('/opt/airflow/data/X_test.csv')['Text']
    
    # Limiter la taille du vocabulaire
    cv = CountVectorizer(max_features=5000)
    cv.fit(X_train)
    
    # Transformer par petits lots
    batch_size = 1000
    x_train_list = []
    for i in range(0, len(X_train), batch_size):
        batch = X_train[i:i+batch_size]
        x_train_list.append(cv.transform(batch).toarray())
    x_train = np.vstack(x_train_list)
    
    x_test = cv.transform(X_test).toarray()
    
    np.save('/opt/airflow/data/x_train.npy', x_train)
    np.save('/opt/airflow/data/x_test.npy', x_test)
    
    with open('/opt/airflow/models/count_vectorizer.pkl', 'wb') as f:
        pickle.dump(cv, f)
    
    print(f"Vocabulary: {len(cv.vocabulary_)}")
    del X_train, X_test, x_train, x_test, x_train_list, cv
    gc.collect()

def train_model(**context):
    x_train = np.load('/opt/airflow/data/x_train.npy')
    y_train = pd.read_csv('/opt/airflow/data/y_train.csv')['Language'].values
    
    print(f"Train shape: {x_train.shape}")
    
    model = MultinomialNB()
    model.fit(x_train, y_train)
    
    with open('/opt/airflow/models/model_latest.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model trained")
    del x_train, y_train, model
    gc.collect()

def evaluate_model(**context):
    x_test = np.load('/opt/airflow/data/x_test.npy')
    y_test = pd.read_csv('/opt/airflow/data/y_test.csv')['Language'].values
    
    model = pickle.load(open('/opt/airflow/models/model_latest.pkl', 'rb'))
    le = pickle.load(open('/opt/airflow/models/label_encoder.pkl', 'rb'))
    
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    pd.DataFrame({
        'Text': pd.read_csv('/opt/airflow/data/X_test.csv')['Text'],
        'actual': [le.classes_[i] for i in y_test],
        'predicted': [le.classes_[i] for i in y_pred],
        'correct': y_test == y_pred
    }).to_csv('/opt/airflow/output/predictions.csv', index=False)
    
    del x_test, y_test, y_pred, model, le
    gc.collect()

def create_pipeline(**context):
    X_train = pd.read_csv('/opt/airflow/data/X_train.csv')['Text']
    y_train = pd.read_csv('/opt/airflow/data/y_train.csv')['Language'].values
    
    cv = pickle.load(open('/opt/airflow/models/count_vectorizer.pkl', 'rb'))
    model = pickle.load(open('/opt/airflow/models/model_latest.pkl', 'rb'))
    
    pipe = Pipeline([('vectorizer', cv), ('multinomialNB', model)])
    
    with open('/opt/airflow/models/trained_pipeline.pkl', 'wb') as f:
        pickle.dump(pipe, f)
    
    print("Pipeline saved")
    del X_train, y_train, cv, model, pipe
    gc.collect()

def test_pipeline(**context):
    pipe = pickle.load(open('/opt/airflow/models/trained_pipeline.pkl', 'rb'))
    le = pickle.load(open('/opt/airflow/models/label_encoder.pkl', 'rb'))
    
    tests = ["Hello, how are you?", "Bonjour", "Hola"]
    
    for text in tests:
        y = pipe.predict([text])
        print(f"{text} → {le.classes_[y[0]]}")
    
    del pipe, le
    gc.collect()

with DAG(
    'language_detection',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['ml', 'nlp'],
) as dag:
    
    t1 = PythonOperator(task_id='load_dataset', python_callable=load_dataset)
    t2 = PythonOperator(task_id='encode_labels', python_callable=encode_labels)
    t3 = PythonOperator(task_id='clean_text', python_callable=clean_text)
    t4 = PythonOperator(task_id='split_data', python_callable=split_data)
    t5 = PythonOperator(task_id='create_bag_of_words', python_callable=create_bag_of_words)
    t6 = PythonOperator(task_id='train_model', python_callable=train_model)
    t7 = PythonOperator(task_id='evaluate_model', python_callable=evaluate_model)
    t8 = PythonOperator(task_id='create_pipeline', python_callable=create_pipeline)
    t9 = PythonOperator(task_id='test_pipeline', python_callable=test_pipeline)
    
    t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7 >> t8 >> t9