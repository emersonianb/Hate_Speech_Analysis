import os
import joblib
import numpy as np
from src.preprocessing import preprocess_tweets

CLASS_LABELS = {
    0: "Discurso de Ódio",
    1: "Linguagem Ofensiva",
    2: "Linguagem Neutra"
}

class HateSpeechPredictor:
    def __init__(self, model_path='models/mlp_model.pkl', vectorizer_path='models/tfidf_vectorizer.pkl'):
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError("Modelo ou vetorizador não encontrados na pasta 'models/'.")
        
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, tweets):
        is_single = isinstance(tweets, str)
        if is_single:
            tweets = [tweets]

        inputs = preprocess_tweets(tweets, self.vectorizer)
        raw_probs = self.model.predict_proba(inputs)
        
        results = []
        for i, probs in enumerate(raw_probs):
            pred_class = int(np.argmax(probs))
            results.append({
                'tweet': tweets[i],
                'label': CLASS_LABELS[pred_class],
                'class_id': pred_class,
                'scores': {
                    'Discurso de Ódio': float(probs[0]),
                    'Linguagem Ofensiva': float(probs[1]),
                    'Linguagem Neutra': float(probs[2])
                }
            })
            
        return results[0] if is_single else results