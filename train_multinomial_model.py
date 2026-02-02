import numpy as np
import pandas as pd
import joblib
import time 
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Models and metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score

def train(data_path="processed_data.pkl", model_path="multinomial_model.pkl", cm_img_path="cm_multinomial.png"):
    """
    Executes the Multinomial Naive Bayes training logic.
    Returns: (model, gui_metrics, predictions)
    """
    
    start_time = time.time() # Start timer
    
    # Load processed data
    print("Cargando datos procesados...")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encuentra el archivo: {data_path}")

    data = joblib.load(data_path)

    # Key adaptation for compatibility
    if 'train_arr' in data:
        train_x = data['train_arr']
        test_x = data['test_arr']
        train_y = data['train_y']
        test_y = data['test_y']
    else:
        # Fallback for short key names
        train_x = data.get('tr_x')
        test_x = data.get('te_x')
        train_y = data.get('tr_y')
        test_y = data.get('te_y')

    print(f"Train shape: {train_x.shape}")
    print(f"Test shape: {test_x.shape}")

    # Initialize Multinomial Model
    multinomial_model = MultinomialNB()

    # --- ORIGINAL DATA LIMITATION LOGIC ---
    # LIMIT is False in the original script
    LIMIT = False
    if LIMIT:
        N = 20000
        train_x_sliced = train_x[20000:N+20000]
        train_y_sliced = train_y[20000:N+20000]
    else:
        train_x_sliced = train_x
        train_y_sliced = train_y

    print("Entrenando modelo...")
    multinomial_model.fit(train_x_sliced, train_y_sliced)

    print("Prediciendo...")
    predictions = multinomial_model.predict(test_x)

    print("Primeras 20 categorias reales:")
    print(test_y.tolist()[:20])
    print("Primeras 20 predicciones:")
    print(predictions.tolist()[:20])

    # --- ORIGINAL EVALUATION ---
    def evaluate_model_internally(y_true, y_pred):
        print("Puntuacion F1:")
        print(f1_score(y_true, y_pred, average='micro'))
        print("Precision:")
        accuracy = accuracy_score(y_true, y_pred)
        print(accuracy)
        print("Precision %:")
        print(round(accuracy*100, 3), "%")
        return accuracy

    acc = evaluate_model_internally(test_y, predictions)

    try:
        rec = recall_score(test_y, predictions, pos_label='spam')
        f1_binary = f1_score(test_y, predictions, pos_label='spam')
    except:
        rec = 0.0
        f1_binary = 0.0
    
    duration_sec = time.time() - start_time
    
    gui_metrics = {
        "accuracy": acc,
        "recall": rec,
        "f1": f1_binary,
        "duracion_seg": round(duration_sec, 2)
    }

    labels = ['ham', 'spam']

    def save_confusion_matrix(y_true, y_pred, color_map, output_path):
        cm = confusion_matrix(y_true, y_pred)
        tick_labels = labels if cm.shape == (2,2) else None
        
        cm_df = pd.DataFrame(cm, index=tick_labels, columns=tick_labels)
        sns.set(font_scale=1.5)
        plt.figure(figsize=(8, 8))

        sns.heatmap(cm_df, cmap=color_map, linewidths=1, annot=True,
                    square=True, fmt='d', cbar=False,
                    xticklabels=tick_labels, yticklabels=tick_labels)

        plt.xlabel("Predicciones")
        plt.ylabel("Realidad")
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()

    # Call with original 'YlGnBu' color
    save_confusion_matrix(test_y, predictions, 'YlGnBu', cm_img_path)

    # Save trained model
    joblib.dump(multinomial_model, model_path)
    print(f"Modelo guardado en '{model_path}'")

    return multinomial_model, gui_metrics, predictions

if __name__ == "__main__":
    train()