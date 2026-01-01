from fastapi import FastAPI, HTTPException, Header, Depends, Security
from fastapi.security import APIKeyHeader
from typing import List
import pickle
import json
from pathlib import Path

app = FastAPI(title="Language Detection API")

# Configuration du schéma de sécurité pour Swagger
api_key_header = APIKeyHeader(name="Authorization", auto_error=True)

# Chemins
TOKENS_FILE = Path("/app/.token")

# Variables globales
pipeline = None
encoder = None
tokens_db = {}


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
    global pipeline, encoder
    
    print("🚀 Démarrage de l'API...")
    
    # Charger les tokens
    load_tokens()
    
    # Charger les modèles
    print("📦 Chargement des modèles...")
    try:
        pipeline = pickle.load(open('/app/models/trained_pipeline.pkl', 'rb'))
        encoder = pickle.load(open('/app/models/label_encoder.pkl', 'rb'))
        print(f"✅ Modèles chargés avec succès")
        print(f"📊 Langues supportées: {list(encoder.classes_)}")
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")


@app.get("/")
def home():
    """Page d'accueil (accès public)"""
    return {
        "service": "Language Detection API",
        "version": "2.0.0",
        "model_loaded": pipeline is not None,
        "languages": list(encoder.classes_) if encoder else [],
        "authentication": "Token requis pour certaines opérations"
    }


@app.post("/predict")
def predict(
    texts: List[str],
    token_info: dict = Depends(verify_token_has_permission("predict"))
):
    """
    Prédire la langue des textes
    
    Requiert un token avec la permission: predict
    """
    if not pipeline or not encoder:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    predictions = pipeline.predict(texts)
    results = []
    
    for text, pred in zip(texts, predictions):
        results.append({
            "text": text,
            "language": encoder.classes_[pred]
        })
    
    return {
        "user_role": token_info["role"],
        "predictions": results
    }


@app.post("/reload")
def reload(token_info: dict = Depends(verify_token_has_permission("reload"))):
    """
    Recharger les modèles
    
    Requiert un token avec la permission: reload
    """
    global pipeline, encoder
    
    try:
        pipeline = pickle.load(open('/app/models/trained_pipeline.pkl', 'rb'))
        encoder = pickle.load(open('/app/models/label_encoder.pkl', 'rb'))
        print(f"🔄 Modèles rechargés par {token_info['role']}")
        return {
            "status": "success",
            "message": "Modèles rechargés avec succès",
            "reloaded_by": token_info["role"],
            "languages": list(encoder.classes_)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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