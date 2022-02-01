"""
File Created: 31th January 2022
Author: Enrique Corimayo 
-----
"""

import pandas as pd
import numpy as np
import datetime as dt
from scipy import stats
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import pickle

# Importamos los datasets
data_train = pd.read_csv("C:/Users/54112/source/repos/ventas_pi/data/Train_BigMart.csv")
data_test = pd.read_csv("C:/Users/54112/source/repos/ventas_pi/data/Test_BigMart.csv")

""""
PRIMER PASO: GUARDAR DATOS ÚTILES
"""

# Este primer paso es importante; como las columnas dummies se obtienen a partir de un dataset entero
# si tenemos solo un fila, con un solo valor por columna, no podemos generar dummies, por lo que
# generaremos las columnas dummieS y guardaremos los nombres para ser usadas mas adelante
x = pd.get_dummies(data_train, columns=["Outlet_Type"])
# Quit Outlet_Type_ from the last columns
x.columns = x.columns.str.replace("Outlet_Type_", "")
columnas = data_train.drop(["Outlet_Type"], axis=1).columns.tolist()
dummies = x.columns.tolist()
for i in columnas:
    dummies.remove(i)
with open("C:/Users/54112/source/repos/ventas_pi/pipeline/dummies.pkl", "wb") as f:
    pickle.dump(dummies, f)
# Identificando la data de train y de test, para posteriormente unión y separación
data_train["Set"] = "train"
data_test["Set"] = "test"

data = pd.concat([data_train, data_test], ignore_index=True, sort=False)
# Los valores nulos de los pesos se completan con la moda del peso de cada Item...
# lo mismo que antes... si tenemos una sola fila no podremos generar esos valores
productos = list(data[data.iloc[:, 1].isnull()].iloc[:, 0].unique())
mode_dict = {}
for i in productos:
    moda = data[data["Item_Identifier"] == i]["Item_Weight"].mode()[0]
    mode_dict[i] = moda
# Para codificar los niveles de precios se usan cuantiles los cuales surgen de usar
# el dataset entero... de nuevo si tenemos una sola fila no podremos generar esos valores
with open("C:/Users/54112/source/repos/ventas_pi/pipeline/mode_dict.pkl", "wb") as f:
    pickle.dump(mode_dict, f)
q1 = data.iloc[:, 5].quantile(0.25)
q2 = data.iloc[:, 5].quantile(0.50)
q3 = data.iloc[:, 5].quantile(0.75)
q = [q1, q2, q3]

with open("C:/Users/54112/source/repos/ventas_pi/pipeline/q.pkl", "wb") as f:
    pickle.dump(q, f)

""""
SEGUNDO PASO: CREACIÓN FUNCTION TRANSFORMER
"""


def ft(x):
    Z = x.copy()
    with open("C:/Users/54112/source/repos/ventas_pi/pipeline/dummies.pkl", "rb") as f:
        dummies = pickle.load(f)
    # FEATURES ENGINEERING: Codificación de variables nominales
    for i in dummies:
        Z[i] = np.where(Z.iloc[:, 10] == i, 1, 0)
    Z.drop(["Outlet_Type"], axis=1, inplace=True)
    # Pasamos todo a numpy para trabajar mas rápido
    Z = Z.to_numpy()
    # FEATURES ENGINEERING: para los años de establecimiento
    Z[:, 7] = 2020 - Z[:, 7]
    # LIMPIEZA: Unificando etiquetas para 'Item_Fat_Content'
    Z[:, 2] = np.where((Z[:, 2] == "low fat") | (Z[:, 2] == "LF"), "Low Fat", Z[:, 2])
    Z[:, 2] = np.where(Z[:, 2] == "reg", "Regular", Z[:, 2])
    # LIMPIEZA: de faltantes en el peso de los productos
    with open(
        "C:/Users/54112/source/repos/ventas_pi/pipeline/mode_dict.pkl", "rb"
    ) as f:
        mode_dict = pickle.load(f)
    for i, j in mode_dict.items():
        Z[:, 1] = np.where(
            (Z[:, 0] == i) & (np.isnan(Z[:, 1].astype(float))), j, Z[:, 1]
        )
    # FEATURES ENGINEERING: Codificando los niveles de precios de los productos
    with open("C:/Users/54112/source/repos/ventas_pi/pipeline/q.pkl", "rb") as f:
        q = pickle.load(f)
    Z[:, 5] = np.where(Z[:, 5] <= q[0], 1, Z[:, 5])
    Z[:, 5] = np.where((Z[:, 5] > q[0]) & (Z[:, 5] <= q[1]), 2, Z[:, 5])
    Z[:, 5] = np.where((Z[:, 5] > q[1]) & (Z[:, 5] <= q[2]), 3, Z[:, 5])
    Z[:, 5] = np.where(Z[:, 5] > q[2], 4, Z[:, 5])
    # FEATURES ENGINEERING: Codificación de variables ordinales
    Z[:, 8] = np.where(Z[:, 8] == "High", 2, Z[:, 8])
    Z[:, 8] = np.where(Z[:, 8] == "Medium", 1, Z[:, 8])
    Z[:, 8] = np.where(Z[:, 8] == "Small", 0, Z[:, 8])

    Z[:, 9] = np.where(Z[:, 9] == "Tier 1", 2, Z[:, 9])
    Z[:, 9] = np.where(Z[:, 9] == "Tier 2", 1, Z[:, 9])
    Z[:, 9] = np.where(Z[:, 9] == "Tier 3", 0, Z[:, 9])
    # LIMPIEZA: de faltantes en el tamaño de las tiendas
    Z[:, 8] = np.where(np.isnan(Z[:, 8].astype(float)), 0, Z[:, 8])

    # Drop Item type e Item_Fat_Content
    Z = np.delete(Z, [2, 4], axis=1)
    # Drop Item Identifier y Outlet Identifier
    Z = np.delete(Z, [0, 4], axis=1)

    return Z


""""
TERCER PASO: CREACIÓN DEL PIPELINE + FITTING
"""

if __name__ == "__main__":

    # X, y de entrenamiento
    X = data_train.drop(columns=["Set", "Item_Outlet_Sales"])
    y = data_train["Item_Outlet_Sales"]
    # División de dataset de entrenaimento y validación
    x_train, x_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=28
    )
    # Creación del pipeline
    pipe = Pipeline([("ft", FunctionTransformer(ft)), ("clf", LinearRegression())])
    pipe.fit(x_train, y_train)
    # Predicción de los valores de validación
    pred = pipe.predict(x_val)

    # Cálculo de los errores cuadráticos medios y Coeficiente de Determinación (R^2)
    mse_train = metrics.mean_squared_error(y_train, pipe.predict(x_train))
    R2_train = pipe.score(x_train, y_train)
    print("Métricas del Modelo:")
    print("ENTRENAMIENTO: RMSE: {:.2f} - R2: {:.4f}".format(mse_train**0.5, R2_train))
    # Cálculo de los errores cuadráticos medios y Coeficiente de Determinación (R^2)
    mse_val = metrics.mean_squared_error(y_val, pred)
    R2_val = pipe.score(x_val, y_val)
    print("VALIDACIÓN: RMSE: {:.2f} - R2: {:.4f}".format(mse_val**0.5, R2_val))
    # Guardamos el pipeline
    with open("C:/Users/54112/source/repos/ventas_pi/pipeline/pipe.pkl", "wb") as f:
        pickle.dump(pipe, f)
