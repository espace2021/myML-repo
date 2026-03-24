# app.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Crée l'app FastAPI
app = FastAPI(title="API Analyse Risque Projet", version="1.0")

# Modèle Pydantic pour valider l'entrée
class Project(BaseModel):
    budget: float
    duree: int
    equipe: int

# Charger le modèle ML
model = joblib.load("model.pkl")

# Endpoint de test simple
@app.get("/")
def home():
    return {"message": "API OK"}

# Endpoint prédiction
@app.post("/predict")
def predict(data: Project):
    # Préparer les données pour le modèle
    X = np.array([[data.budget, data.duree, data.equipe]])

    # Faire la prédiction
    prediction = model.predict(X)

    # Optionnel : calculer les probabilités
    proba = model.predict_proba(X)

    # Retourner un JSON avec la classe de risque et les probabilités
    return {
        "risk_class": int(prediction[0]),          # 0 = faible, 1 = élevé
        "risk_probability": proba[0].tolist()      # probabilités [faible, élevé]
    }