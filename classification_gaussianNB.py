from sklearn.naive_bayes import GaussianNB
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

def naive_bayes_classification(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_texts, train_labels = zip(*train_data)
    test_texts, test_labels = zip(*test_data)
    
    train_tokens = [text.split() for text in train_texts]
    test_tokens = [text.split() for text in test_texts]
    
    # Train Word2Vec model
    vectorizer = Word2Vec(sentences=train_tokens, vector_size=100, window=5, min_count=1, workers=2)
    vectorizer.build_vocab(train_tokens)
    vectorizer.train(train_tokens, total_examples=len(train_tokens), epochs=10)
    
    
    def get_doc_vector(tokens):
        vectors = [vectorizer.wv[word] for word in tokens if word in vectorizer.wv]
        if len(vectors) > 0:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(vectorizer.vector_size)
    
    train_vectors = np.array([get_doc_vector(tokens) for tokens in train_tokens])
    test_vectors = np.array([get_doc_vector(tokens) for tokens in test_tokens])
    
    model = GaussianNB()
    model.fit(train_vectors, train_labels)
    predictions = model.predict(test_vectors)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy 