import numpy as np
import pandas as pd
import joblib
import time # Necesario para medir el tiempo en la App
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Modelos y métricas
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score

def entrenar(path_datos="datos_procesados.pkl", path_modelo="modelo_gaussiano.pkl", path_img_cm="cm_gaussiano.png"):
    """
    Ejecuta la lógica de entrenamiento del Naive Bayes Gaussiano.
    Devuelve: (modelo, metricas_para_gui, predicciones)
    """
    
    start_time = time.time() # Inicio cronómetro
    
    # Cargar datos procesados
    print("Cargando datos procesados...")
    
    if not os.path.exists(path_datos):
        raise FileNotFoundError(f"No se encuentra el archivo: {path_datos}")

    data = joblib.load(path_datos)

    # Adaptación de claves (Compatibilidad con tu script y la App)
    if 'train_arr' in data:
        train_arr = data['train_arr']
        test_arr = data['test_arr']
        train_y = data['train_y']
        test_y = data['test_y']
    else:
        # Fallback si el pickle viene con nombres cortos
        train_arr = data.get('tr_x')
        test_arr = data.get('te_x')
        train_y = data.get('tr_y')
        test_y = data.get('te_y')

    print(f"Train shape: {train_arr.shape}")
    print(f"Test shape: {test_arr.shape}")

    # Entrenar modelo GaussianNB
    modeloGausiano = GaussianNB()

    # --- LÓGICA ORIGINAL DE LIMITACIÓN DE DATOS ---
    LIMITAR = False
    if LIMITAR:
        N = 20000
        # Slice exacto que tenías:
        train_arr_cortado = train_arr[20000:N+20000]
        train_y_cortado = train_y[20000:N+20000]
    else:
        train_arr_cortado = train_arr
        train_y_cortado = train_y

    print("Entrenando modelo...")
    modeloGausiano.fit(train_arr_cortado, train_y_cortado)

    print("Prediciendo...")
    pred = modeloGausiano.predict(test_arr)

    print("Primeras 20 categorias reales:")
    print(test_y.tolist()[:20])
    print("Primeras 20 predicciones:")
    print(pred.tolist()[:20])

    # --- EVALUACIÓN ORIGINAL ---
    def evaluar_modelo_interno(y, y_pred):
        print("Puntuacion F1:")
        # Tu lógica original usa 'micro'
        print(f1_score(y, y_pred, average='micro'))
        print("Precision:")
        acc = accuracy_score(y, y_pred)
        print(acc)
        print("Precision %:")
        print(round(acc*100, 3), "%")
        return acc # Retornamos para usarlo fuera

    acc = evaluar_modelo_interno(test_y, pred)

    try:
        rec = recall_score(test_y, pred, pos_label='spam')
        f1_binary = f1_score(test_y, pred, pos_label='spam')
    except:
        rec = 0.0
        f1_binary = 0.0
    
    duracion = time.time() - start_time
    
    metrics_gui = {
        "accuracy": acc,
        "recall": rec,
        "f1": f1_binary,
        "duracion_seg": round(duracion, 2)
    }

    etiquetas = ['ham', 'spam'] #

    def matriz_confusion_guardar(test_y, pred, color, ruta_salida):
        cof = confusion_matrix(test_y, pred)
        
        # Validar dimensiones para evitar error si solo hay 1 clase predicha
        lbls = etiquetas if cof.shape == (2,2) else None

        cof_df = pd.DataFrame(cof, index=lbls, columns=lbls)
        sns.set(font_scale=1.5)
        plt.figure(figsize=(8, 8))

        sns.heatmap(cof_df, cmap=color, linewidths=1, annot=True,
                    square=True, fmt='d', cbar=False,
                    xticklabels=lbls, yticklabels=lbls)

        plt.xlabel("Predicciones")
        plt.ylabel("Realidad")
        
        # CAMBIO: Savefig en vez de show para no bloquear la app
        if ruta_salida:
            plt.savefig(ruta_salida)
            plt.close() # Liberar memoria
        else:
            plt.show()

    # Llamamos a la función con tu color original 'YlGnBu'
    matriz_confusion_guardar(test_y, pred, 'YlGnBu', path_img_cm)

    # Guardar modelo entrenado
    joblib.dump(modeloGausiano, path_modelo)
    print(f"Modelo guardado en '{path_modelo}'")
    
    return modeloGausiano, metrics_gui, pred

if __name__ == "__main__":
    entrenar()