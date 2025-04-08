from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

def naive_bayes_classification(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_texts, train_labels = zip(*train_data)
    test_texts, test_labels = zip(*test_data)

    # Convert text data to feature vectors
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()

    # Train Multinomial Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train, train_labels)

    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    return accuracy