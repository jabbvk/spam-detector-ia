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

def train(data_path="processed_data.pkl", model_path="tree_model.pkl", cm_img_path="cm_tree.png"):
    start_time = time.time()
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encuentra el archivo: {data_path}")
        
    data = joblib.load(data_path)

    # Safe data loading (fixes the 'ambiguous' error)
    if 'train_arr' in data:
        train_x = data['train_arr']
        test_x = data['test_arr']
        train_y = data['train_y']
        test_y = data['test_y']
    else:
        train_x = data['tr_x']
        test_x = data['te_x']
        train_y = data['tr_y']
        test_y = data['te_y']

    tree_model = DecisionTreeClassifier()

    # Ensure LIMIT is False
    LIMIT = False
    if LIMIT:
        N = 20000
        train_x_sliced = train_x[20000:N+20000]
        train_y_sliced = train_y[20000:N+20000]
    else:
        train_x_sliced = train_x
        train_y_sliced = train_y

    tree_model.fit(train_x_sliced, train_y_sliced)
    predictions = tree_model.predict(test_x)

    # Metrics calculation
    acc = accuracy_score(test_y, predictions)
    try:
        rec = recall_score(test_y, predictions, pos_label='spam')
        f1_bin = f1_score(test_y, predictions, pos_label='spam')
    except:
        rec = 0.0
        f1_bin = 0.0

    gui_metrics = {
        "accuracy": acc,
        "recall": rec,
        "f1": f1_bin,
        "duracion_seg": round(time.time() - start_time, 2)
    }

    # Confusion Matrix
    labels = ['ham', 'spam']
    cm = confusion_matrix(test_y, predictions)
    plt.figure(figsize=(8, 8))
    sns.heatmap(pd.DataFrame(cm, index=labels, columns=labels), 
                cmap='YlGnBu', annot=True, fmt='d', cbar=False)
    plt.xlabel("Predicciones")
    plt.ylabel("Realidad")
    plt.savefig(cm_img_path)
    plt.close()

    joblib.dump(tree_model, model_path)
    
    return tree_model, gui_metrics, predictions

if __name__ == "__main__":
    train()