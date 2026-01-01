import numpy as np
import pickle
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

tf.config.run_functions_eagerly(True)

VOCAB_SIZE = 5000
MAX_LENGTH = 50

# Charger les composants
model = tf.keras.models.load_model('./models/dl_model.h5')
with open('./models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('./models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Fonction de prédiction
def predict(text):
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', text).lower().strip()
    sequence = tokenizer.texts_to_sequences([text])
    sequence = [[min(idx, VOCAB_SIZE-1) for idx in seq] for seq in sequence]
    padded = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post')
    prediction = model.predict(padded, verbose=0)
    lang = le.classes_[np.argmax(prediction)]
    conf = np.max(prediction)
    return lang, conf

# Tests
tests = [
    "Hello, how are you?",
    "Bonjour tout le monde",
    "Hola mundo cómo estás",
    "Hallo, wie geht's dir?",]

print("Tests de détection de langue:")
print("=" * 60)
for text in tests:
    lang, conf = predict(text)
    print(f"{text:30s} → {lang:10s} ({conf:.2%})")