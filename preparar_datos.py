import os
import numpy as np
import pandas as pd
import string as s
import joblib

import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

def tokenizar(texto):
    tokens = texto.split()
    return tokens

def hacer_minusculas(tokens):
    nuevos_tokens=[]
    for t in tokens:
        t = t.lower()
        nuevos_tokens.append(t)
    return nuevos_tokens

def eliminar_simbolos(tokens):
    nuevos_tokens = []
    for t in tokens:
        for simbolo in s.punctuation:
            if simbolo != "'":
                t = t.replace(simbolo, '')
        nuevos_tokens.append(t)
    return nuevos_tokens

def eliminar_numeros(tokens):
    sin_numeros = []
    nuevos_tokens = []
    for t in tokens:
        for digito in s.digits:
            t = t.replace(digito, '')
        sin_numeros.append(t)
    for t in sin_numeros:
        if t != '':
            nuevos_tokens.append(t)
    return nuevos_tokens

def quitar_invariantes(tokens):
    invariables = stopwords.words('english')
    nuevos_tokens = []
    for t in tokens:
        if t not in invariables:
            nuevos_tokens.append(t)
    return nuevos_tokens

lematizador = nltk.stem.WordNetLemmatizer()

def lematizar(tokens):
    nuevos_tokens = []
    for t in tokens:
        t = lematizador.lemmatize(t)
        nuevos_tokens.append(t)
    return nuevos_tokens


def ejecutar_preparacion(path_train, path_test, path_salida_pkl):
    """
    Ejecuta toda la lógica de preprocesamiento.
    Args:
        path_train: Ruta al CSV de entrenamiento
        path_test: Ruta al CSV de test
        path_salida_pkl: Ruta donde guardar el .pkl
    Returns:
        (exito: bool, mensaje: str)
    """
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

        print(f"Cargando datos desde: {path_train}")
        
        if not os.path.exists(path_train):
            return False, "No se encuentra el archivo spam.csv"

        df = pd.read_csv(path_train, encoding='latin-1')

        df = df[['v1', 'v2']]
        df.columns = ['categoria', 'texto']

        entrenamiento = df.sample(frac=0.8, random_state=42)
        prueba = df.drop(entrenamiento.index)

        print("Primeras filas cargadas:")
        print(entrenamiento.head())

        train_x = entrenamiento.texto
        test_x = prueba.texto
        train_y = entrenamiento.categoria
        test_y = prueba.categoria

        print("Tokenizando...")
        train_x = train_x.apply(tokenizar)
        test_x = test_x.apply(tokenizar)

        print("Convirtiendo a minúsculas...")
        train_x = train_x.apply(hacer_minusculas)
        test_x = test_x.apply(hacer_minusculas)

        print("Eliminando símbolos...")
        train_x = train_x.apply(eliminar_simbolos)
        test_x = test_x.apply(eliminar_simbolos)

        print("Eliminando números...")
        train_x = train_x.apply(eliminar_numeros)
        test_x = test_x.apply(eliminar_numeros)

        print("Eliminando stopwords...")
        train_x = train_x.apply(quitar_invariantes)
        test_x = test_x.apply(quitar_invariantes)

        print("Lematizando...")
        train_x = train_x.apply(lematizar)
        test_x = test_x.apply(lematizar)

        print("Vectorizando...")
        train_x = train_x.apply(lambda x: ''.join(i+' ' for i in x))
        test_x = test_x.apply(lambda x: ''.join(i+' ' for i in x))

        vectorizador = TfidfVectorizer(max_features=5000, min_df=6)
        vectorizador.fit(train_x)

        train_arr = vectorizador.transform(train_x).toarray()
        test_arr = vectorizador.transform(test_x).toarray()

        print(f"Guardando datos procesados en {path_salida_pkl}...")
        
        datos_para_guardar = {
            'train_arr': train_arr,
            'test_arr': test_arr,
            'train_y': train_y,
            'test_y': test_y,
            'vectorizador': vectorizador
        }
        
        joblib.dump(datos_para_guardar, path_salida_pkl)

        print("Datos guardados correctamente.")
        print(f"Train shape: {train_arr.shape}")
        print(f"Test shape: {test_arr.shape}")
        
        return True, "Preprocesamiento y vectorización completados."

    except Exception as e:
        print(f"Error detallado: {e}")
        return False, str(e)

if __name__ == "__main__":
    ruta_dir = os.path.dirname(__file__)
    ejecutar_preparacion(
        os.path.join(ruta_dir, "SuicideTrain.csv"),
        os.path.join(ruta_dir, "SuicideTest.csv"),
        os.path.join(ruta_dir, "datos_procesados.pkl")
    )