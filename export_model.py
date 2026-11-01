import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

def train_and_export():
    os.makedirs('models', exist_ok=True)
    
    # 1. Carregar dataset do data.world
    url = 'https://query.data.world/s/3w2odrweyv6zurbtbmxsxshcuwpwn3?dws=00000'
    print("A descarregar o dataset...")
    df = pd.read_csv(url)
    
    # 2. Limpeza de texto
    df['tweet_clean'] = (
        df['tweet']
        .astype(str)
        .str.replace(r'[^A-Za-z\s]', '', regex=True)
        .str.lower()
        .str.strip()
    )
    
    X_text = df['tweet_clean'].values
    y = df['class'].values
    
    # 3. Vetorização TF-IDF
    print("Vetorização TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(X_text)
    
    # 4. Divisão estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Treino da Rede Neuronal (MLPClassifier: 128 e 64 neurónios)
    print("A treinar a Rede Neuronal...")
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        max_iter=25,
        random_state=42,
        verbose=True
    )
    model.fit(X_train, y_train)
    
    print(f"Precisão no conjunto de teste: {model.score(X_test, y_test)*100:.2f}%")
    
    # 6. Guardar os artefactos
    joblib.dump(model, 'models/mlp_model.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    print("Sucesso: 'models/mlp_model.pkl' e 'models/tfidf_vectorizer.pkl' foram gerados.")

if __name__ == '__main__':
    train_and_export()