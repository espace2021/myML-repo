import joblib
import pandas as pd

model_loaded = joblib.load("model.pkl")

data = pd.DataFrame([{
    "Budget": 10000,
    "Montant_collecte": 3000
}])

prediction = model_loaded.predict(data)

print(prediction)