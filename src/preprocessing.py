import re

def clean_text(text: str) -> str:
    """Aplica a limpeza de texto idêntica ao treinamento."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^A-Za-z\s]', '', text)
    return text.lower().strip()

def preprocess_tweets(tweets, vectorizer):
    """Limpa e vetoriza a lista de textos usando o TF-IDF ajustado."""
    if isinstance(tweets, str):
        tweets = [tweets]
    
    cleaned_tweets = [clean_text(t) for t in tweets]
    tfidf_matrix = vectorizer.transform(cleaned_tweets).toarray()
    return tfidf_matrix