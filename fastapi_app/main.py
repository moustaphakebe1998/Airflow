from fastapi import FastAPI, HTTPException, Header, Depends, Security
from fastapi.security import APIKeyHeader
from typing import List, Optional
import pickle
import json
import re
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI(title="Language Detection API - ML & Deep Learning")

# Configuration du schéma de sécurité pour Swagger
api_key_header = APIKeyHeader(name="Authorization", auto_error=True)

# Chemins
TOKENS_FILE = Path("/app/.token")
ML_PIPELINE_PATH = Path("/app/models/trained_pipeline.pkl")
ML_ENCODER_PATH = Path("/app/models/label_encoder.pkl")
DL_MODEL_PATH = Path("/app/models/dl_model.h5")
DL_TOKENIZER_PATH = Path("/app/models/tokenizer.pkl")
DL_ENCODER_PATH = Path("/app/models/label_encoder.pkl")

# Constantes pour Deep Learning
VOCAB_SIZE = 5000
MAX_LENGTH = 50

# Variables globales
ml_pipeline = None
ml_encoder = None
dl_model = None
dl_tokenizer = None
dl_encoder = None
tokens_db = {}

# Configuration TensorFlow
tf.config.run_functions_eagerly(True)


def load_tokens():
    """Charger les tokens depuis le fichier .token"""
    global tokens_db
    
    if not TOKENS_FILE.exists():
        print("⚠️  Fichier .token non trouvé, création avec token par défaut...")
        default_tokens = {
            "adminkebe": {
                "role": "admin",
                "permissions": ["reload", "predict", "manage"],
                "description": "Administrateur principal"
            }
        }
        with open(TOKENS_FILE, 'w') as f:
            json.dump(default_tokens, f, indent=2)
        tokens_db = default_tokens
    else:
        with open(TOKENS_FILE, 'r') as f:
            tokens_db = json.load(f)
    
    print(f"🔑 {len(tokens_db)} token(s) chargé(s)")
    for token, info in tokens_db.items():
        print(f"   - {token}: {info['role']} ({info['description']})")


def verify_token_has_permission(required_permission: str):
    """Retourne une fonction de vérification pour une permission spécifique"""
    def verify(authorization: str = Security(api_key_header)):
        if authorization not in tokens_db:
            raise HTTPException(
                status_code=401,
                detail="Token invalide"
            )
        
        token_info = tokens_db[authorization]
        
        if required_permission not in token_info["permissions"]:
            raise HTTPException(
                status_code=403,
                detail=f"Permission '{required_permission}' requise. Votre rôle: {token_info['role']}"
            )
        
        return token_info
    
    return verify


def verify_any_token(authorization: str = Security(api_key_header)):
    """Vérifier que le token existe (sans vérifier les permissions)"""
    if authorization not in tokens_db:
        raise HTTPException(
            status_code=401,
            detail="Token invalide"
        )
    return tokens_db[authorization]


@app.on_event("startup")
def startup():
    """Initialisation au démarrage"""
    global ml_pipeline, ml_encoder, dl_model, dl_tokenizer, dl_encoder
    
    print("🚀 Démarrage de l'API...")
    
    # Charger les tokens
    load_tokens()
    
    # Charger le modèle ML classique
    print("\n📦 Chargement du modèle ML classique...")
    try:
        if ML_PIPELINE_PATH.exists() and ML_ENCODER_PATH.exists():
            ml_pipeline = pickle.load(open(ML_PIPELINE_PATH, 'rb'))
            ml_encoder = pickle.load(open(ML_ENCODER_PATH, 'rb'))
            print(f"✅ Modèle ML chargé")
            print(f"📊 Langues ML: {list(ml_encoder.classes_)}")
        else:
            print("⚠️  Modèle ML non trouvé")
    except Exception as e:
        print(f"❌ Erreur ML: {e}")
    
    # Charger le modèle Deep Learning
    print("\n📦 Chargement du modèle Deep Learning...")
    try:
        if DL_MODEL_PATH.exists() and DL_TOKENIZER_PATH.exists() and DL_ENCODER_PATH.exists():
            dl_model = tf.keras.models.load_model(str(DL_MODEL_PATH))
            dl_tokenizer = pickle.load(open(DL_TOKENIZER_PATH, 'rb'))
            dl_encoder = pickle.load(open(DL_ENCODER_PATH, 'rb'))
            print(f"✅ Modèle DL chargé")
            print(f"📊 Langues DL: {list(dl_encoder.classes_)}")
        else:
            print("⚠️  Modèle DL non trouvé")
    except Exception as e:
        print(f"❌ Erreur DL: {e}")
    
    print("\n✅ API prête!")


def predict_ml(texts: List[str]):
    """Prédire avec le modèle ML classique"""
    if not ml_pipeline or not ml_encoder:
        raise HTTPException(status_code=503, detail="Modèle ML non chargé")
    
    predictions = ml_pipeline.predict(texts)
    results = []
    
    for text, pred in zip(texts, predictions):
        results.append({
            "text": text,
            "language": ml_encoder.classes_[pred],
            "model": "ML"
        })
    
    return results


def predict_dl(texts: List[str]):
    """Prédire avec le modèle Deep Learning"""
    if not dl_model or not dl_tokenizer or not dl_encoder:
        raise HTTPException(status_code=503, detail="Modèle DL non chargé")
    
    results = []
    
    for text in texts:
        # Nettoyer le texte
        cleaned_text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).lower().strip()
        
        # Tokenizer et padding
        sequence = dl_tokenizer.texts_to_sequences([cleaned_text])
        
        if not sequence[0]:
            results.append({
                "text": text,
                "language": "unknown",
                "confidence": 0.0,
                "model": "DL"
            })
            continue
        
        # Limiter les indices
        sequence = [[min(idx, VOCAB_SIZE-1) for idx in seq] for seq in sequence]
        padded = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post')
        
        # Prédiction
        prediction = dl_model.predict(padded, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction, axis=1)[0]
        
        results.append({
            "text": text,
            "language": dl_encoder.classes_[predicted_class],
            "confidence": float(confidence),
            "model": "DL"
        })
    
    return results


@app.get("/")
def home():
    """Page d'accueil (accès public)"""
    return {
        "service": "Language Detection API",
        "version": "3.0.0",
        "models": {
            "ml": {
                "loaded": ml_pipeline is not None,
                "languages": list(ml_encoder.classes_) if ml_encoder else []
            },
            "dl": {
                "loaded": dl_model is not None,
                "languages": list(dl_encoder.classes_) if dl_encoder else []
            }
        },
        "authentication": "Token requis pour certaines opérations",
        "endpoints": {
            "/predict/ml": "Prédiction avec modèle ML classique",
            "/predict/dl": "Prédiction avec modèle Deep Learning",
            "/predict/both": "Prédiction avec les deux modèles"
        }
    }


@app.post("/predict/ml")
def predict_ml_endpoint(
    texts: List[str],
    token_info: dict = Depends(verify_token_has_permission("predict"))
):
    """
    Prédire la langue avec le modèle ML classique
    
    Requiert un token avec la permission: predict
    """
    results = predict_ml(texts)
    
    return {
        "user_role": token_info["role"],
        "model": "ML",
        "predictions": results
    }


@app.post("/predict/dl")
def predict_dl_endpoint(
    texts: List[str],
    token_info: dict = Depends(verify_token_has_permission("predict"))
):
    """
    Prédire la langue avec le modèle Deep Learning
    
    Requiert un token avec la permission: predict
    """
    results = predict_dl(texts)
    
    return {
        "user_role": token_info["role"],
        "model": "Deep Learning",
        "predictions": results
    }


@app.post("/predict/both")
def predict_both(
    texts: List[str],
    token_info: dict = Depends(verify_token_has_permission("predict"))
):
    """
    Prédire la langue avec les deux modèles et comparer
    
    Requiert un token avec la permission: predict
    """
    ml_results = predict_ml(texts) if ml_pipeline else []
    dl_results = predict_dl(texts) if dl_model else []
    
    # Combiner les résultats
    combined = []
    for i, text in enumerate(texts):
        comparison = {
            "text": text,
            "ml_prediction": ml_results[i]["language"] if ml_results else "N/A",
            "dl_prediction": dl_results[i]["language"] if dl_results else "N/A",
            "dl_confidence": dl_results[i]["confidence"] if dl_results else 0.0,
            "agreement": ml_results[i]["language"] == dl_results[i]["language"] if (ml_results and dl_results) else False
        }
        combined.append(comparison)
    
    return {
        "user_role": token_info["role"],
        "predictions": combined
    }


@app.post("/predict")
def predict_default(
    texts: List[str],
    model: Optional[str] = "dl",
    token_info: dict = Depends(verify_token_has_permission("predict"))
):
    """
    Prédire la langue (modèle par défaut: DL)
    
    Requiert un token avec la permission: predict
    
    Paramètres:
    - model: "ml" ou "dl" (défaut: "dl")
    """
    if model.lower() == "ml":
        results = predict_ml(texts)
    else:
        results = predict_dl(texts)
    
    return {
        "user_role": token_info["role"],
        "model": model.upper(),
        "predictions": results
    }


@app.post("/reload")
def reload(token_info: dict = Depends(verify_token_has_permission("reload"))):
    """
    Recharger tous les modèles
    
    Requiert un token avec la permission: reload
    """
    global ml_pipeline, ml_encoder, dl_model, dl_tokenizer, dl_encoder
    
    results = {"ml": "not_loaded", "dl": "not_loaded"}
    
    # Recharger ML
    try:
        if ML_PIPELINE_PATH.exists() and ML_ENCODER_PATH.exists():
            ml_pipeline = pickle.load(open(ML_PIPELINE_PATH, 'rb'))
            ml_encoder = pickle.load(open(ML_ENCODER_PATH, 'rb'))
            results["ml"] = "success"
            print(f"🔄 Modèle ML rechargé")
    except Exception as e:
        results["ml"] = f"error: {str(e)}"
    
    # Recharger DL
    try:
        if DL_MODEL_PATH.exists() and DL_TOKENIZER_PATH.exists() and DL_ENCODER_PATH.exists():
            dl_model = tf.keras.models.load_model(str(DL_MODEL_PATH))
            dl_tokenizer = pickle.load(open(DL_TOKENIZER_PATH, 'rb'))
            dl_encoder = pickle.load(open(DL_ENCODER_PATH, 'rb'))
            results["dl"] = "success"
            print(f"🔄 Modèle DL rechargé")
    except Exception as e:
        results["dl"] = f"error: {str(e)}"
    
    return {
        "status": "completed",
        "reloaded_by": token_info["role"],
        "results": results
    }


@app.post("/tokens/reload")
def reload_tokens(token_info: dict = Depends(verify_token_has_permission("manage"))):
    """
    Recharger les tokens depuis le fichier .token
    
    Requiert un token avec la permission: manage
    """
    load_tokens()
    return {
        "status": "success",
        "message": "Tokens rechargés",
        "total_tokens": len(tokens_db)
    }


@app.get("/tokens/list")
def list_tokens(token_info: dict = Depends(verify_token_has_permission("manage"))):
    """
    Lister tous les tokens
    
    Requiert un token avec la permission: manage
    """
    tokens_list = []
    for token, info in tokens_db.items():
        tokens_list.append({
            "token": token[:4] + "..." + token[-4:],
            "role": info["role"],
            "permissions": info["permissions"],
            "description": info["description"]
        })
    
    return {
        "total": len(tokens_list),
        "tokens": tokens_list
    }


@app.get("/me")
def me(token_info: dict = Depends(verify_any_token)):
    """Obtenir les informations de son token"""
    return {
        "role": token_info["role"],
        "permissions": token_info["permissions"],
        "description": token_info["description"]
    }


@app.get("/health")
def health():
    """Vérifier l'état de santé de l'API"""
    return {
        "status": "healthy",
        "models": {
            "ml_loaded": ml_pipeline is not None,
            "dl_loaded": dl_model is not None
        }
    }