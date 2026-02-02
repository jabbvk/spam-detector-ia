import os
import numpy as np
import pandas as pd
import string as s
import joblib

# Natural Language Toolkit
import nltk
from nltk.corpus import stopwords

# Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

def tokenize(text):
    """Splits the text into individual tokens."""
    tokens = text.split()
    return tokens

def to_lowercase(tokens):
    """Converts all tokens in a list to lowercase."""
    new_tokens = []
    for t in tokens:
        t = t.lower()
        new_tokens.append(t)
    return new_tokens

def remove_symbols(tokens):
    """Removes punctuation symbols from tokens, keeping apostrophes."""
    new_tokens = []
    for t in tokens:
        for symbol in s.punctuation:
            if symbol != "'":
                t = t.replace(symbol, '')
        new_tokens.append(t)
    return new_tokens

def remove_numbers(tokens):
    """Removes all digits from the token list."""
    no_numbers = []
    new_tokens = []
    for t in tokens:
        for digit in s.digits:
            t = t.replace(digit, '')
        no_numbers.append(t)
    for t in no_numbers:
        if t != '':
            new_tokens.append(t)
    return new_tokens

def remove_stopwords(tokens):
    """Filters out English stop words from the token list."""
    stop_words = stopwords.words('english')
    new_tokens = []
    for t in tokens:
        if t not in stop_words:
            new_tokens.append(t)
    return new_tokens

# Initialize the lemmatizer
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize(tokens):
    """Reduces words to their base or dictionary form."""
    new_tokens = []
    for t in tokens:
        t = lemmatizer.lemmatize(t)
        new_tokens.append(t)
    return new_tokens


def run_preparation(train_path, test_path, output_pkl_path):
    """
    Executes the full preprocessing logic.
    Args:
        train_path: Path to the training CSV
        test_path: Path to the test CSV
        output_pkl_path: Path where the .pkl will be saved
    Returns:
        (success: bool, message: str)
    """
    try:
        # Download NLTK resources
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

        print(f"Cargando datos desde: {train_path}")
        
        if not os.path.exists(train_path):
            return False, "No se encuentra el archivo spam.csv"

        # Load data using latin-1 encoding for compatibility
        df = pd.read_csv(train_path, encoding='latin-1')

        # Select relevant columns and rename them
        df = df[['v1', 'v2']]
        df.columns = ['categoria', 'texto']

        # Split data into 80% training and 20% testing
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)

        print("Primeras filas cargadas:")
        print(train_df.head())

        train_x = train_df.texto
        test_x = test_df.texto
        train_y = train_df.categoria
        test_y = test_df.categoria

        # Apply preprocessing steps
        print("Tokenizando...")
        train_x = train_x.apply(tokenize)
        test_x = test_x.apply(tokenize)

        print("Convirtiendo a minúsculas...")
        train_x = train_x.apply(to_lowercase)
        test_x = test_x.apply(to_lowercase)

        print("Eliminando símbolos...")
        train_x = train_x.apply(remove_symbols)
        test_x = test_x.apply(remove_symbols)

        print("Eliminando números...")
        train_x = train_x.apply(remove_numbers)
        test_x = test_x.apply(remove_numbers)

        print("Eliminando stopwords...")
        train_x = train_x.apply(remove_stopwords)
        test_x = test_x.apply(remove_stopwords)

        print("Lematizando...")
        train_x = train_x.apply(lemmatize)
        test_x = test_x.apply(lemmatize)

        # Vectorization
        print("Vectorizando...")
        # Join tokens back into strings for TfidfVectorizer
        train_x = train_x.apply(lambda x: ''.join(i+' ' for i in x))
        test_x = test_x.apply(lambda x: ''.join(i+' ' for i in x))

        vectorizer = TfidfVectorizer(max_features=5000, min_df=6)
        vectorizer.fit(train_x)

        train_arr = vectorizer.transform(train_x).toarray()
        test_arr = vectorizer.transform(test_x).toarray()

        print(f"Guardando datos procesados en {output_pkl_path}...")
        
        data_to_save = {
            'train_arr': train_arr,
            'test_arr': test_arr,
            'train_y': train_y,
            'test_y': test_y,
            'vectorizador': vectorizer
        }
        
        joblib.dump(data_to_save, output_pkl_path)

        print("Datos guardados correctamente.")
        print(f"Train shape: {train_arr.shape}")
        print(f"Test shape: {test_arr.shape}")
        
        return True, "Preprocesamiento y vectorización completados."

    except Exception as e:
        print(f"Error detallado: {e}")
        return False, str(e)

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    # Default execution for local testing
    run_preparation(
        os.path.join(current_dir, "spam.csv"),
        os.path.join(current_dir, "spam.csv"),
        os.path.join(current_dir, "processed_data.pkl")
    )