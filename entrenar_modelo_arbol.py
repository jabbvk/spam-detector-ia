import numpy as np
import pandas as pd
import joblib
import time 
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score

def entrenar(path_datos="datos_procesados.pkl", path_modelo="modelo_arbol.pkl", path_img_cm="cm_arbol.png"):
    """
    Ejecuta la lógica de entrenamiento del Árbol de Decisión.
    Devuelve: (modelo_entrenado, diccionario_metricas, predicciones)
    """
    
    start_time = time.time()
    
    print("Cargando datos procesados...")
    
    if not os.path.exists(path_datos):
        raise FileNotFoundError(f"No se encuentra el archivo: {path_datos}")
        
    data = joblib.load(path_datos)
    if 'train_arr' in data:
        train_arr = data['train_arr']
        test_arr = data['test_arr']
        train_y = data['train_y']
        test_y = data['test_y']
    else:
        train_arr = data.get('tr_x')
        test_arr = data.get('te_x')
        train_y = data.get('tr_y')
        test_y = data.get('te_y')

    print(f"Train shape: {train_arr.shape}")
    print(f"Test shape: {test_arr.shape}")

    arbol = DecisionTreeClassifier()

    LIMITAR = False
    if LIMITAR:
        N = 20000
        train_arr_cortado = train_arr[20000:N+20000]
        train_y_cortado = train_y[20000:N+20000]
    else:
        train_arr_cortado = train_arr
        train_y_cortado = train_y

    print("Entrenando modelo...")
    arbol.fit(train_arr_cortado, train_y_cortado)

    print("Prediciendo...")
    pred = arbol.predict(test_arr)

    print("Primeras 20 categorias reales:")
    print(test_y.tolist()[:20])
    print("Primeras 20 predicciones:")
    print(pred.tolist()[:20])

    print("Puntuacion F1 (Micro):")
    f1_micro = f1_score(test_y, pred, average='micro')
    print(f1_micro)
    
    print("Precision:")
    acc = accuracy_score(test_y, pred)
    print(acc)
    
    print("Precision %:")
    print(round(acc*100, 3), "%")

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
    
    cof = confusion_matrix(test_y, pred)
    if cof.shape == (2,2):
        lbls = etiquetas
    else:
        lbls = None

    cof_df = pd.DataFrame(cof, index=lbls, columns=lbls)
    sns.set(font_scale=1.5)
    plt.figure(figsize=(8, 8))

    # Color hardcodeado 'YlGnBu' como en tu script
    sns.heatmap(cof_df, cmap='YlGnBu', linewidths=1, annot=True,
                square=True, fmt='d', cbar=False,
                xticklabels=lbls, yticklabels=lbls)

    plt.xlabel("Predicciones")
    plt.ylabel("Realidad")
    
    # EN LUGAR DE SHOW, GUARDAMOS SI HAY RUTA
    if path_img_cm:
        plt.savefig(path_img_cm)
        plt.close() # Importante cerrar para liberar memoria en la App
    else:
        plt.show()

    # Guardar modelo entrenado
    joblib.dump(arbol, path_modelo)
    print(f"Modelo guardado en '{path_modelo}'")
    
    return arbol, metrics_gui, pred

# Bloque para que se pueda seguir ejecutando solo si quieres
if __name__ == "__main__":
    entrenar()