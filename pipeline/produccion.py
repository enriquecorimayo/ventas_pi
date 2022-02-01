"""
File Created: 31th January 2022
Author: Enrique Corimayo 
-----
"""
import pandas as pd
import pickle

# Importamos la función transformer para poder usarla en el pipeline
from pipeline import ft

# Leemos el archivo json "example.json"
import json

with open("../ventas_pi/notebook/example.json", "rb") as f:
    file = json.load(f)
df = pd.DataFrame([file])
# Abrimos el pipeline
with open("../ventas_pi/pipeline/pipe.pkl", "rb") as f:
    pipe = pickle.load(f)
# Predicción
print(pipe.predict(df).round(2))
