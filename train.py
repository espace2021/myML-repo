import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Charger données
df = pd.read_csv("./dataset_projets_risque.csv")

# Exemple simple (à adapter)
X = df[["Budget", "Montant_collecte"]]
y = (df["Montant_collecte"] > df["Budget"] * 0.8).astype(int)

# Modèle
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Sauvegarde
joblib.dump(model, "model.pkl")

print("Modèle entraîné et sauvegardé !")