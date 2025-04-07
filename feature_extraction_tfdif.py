from sklearn.feature_extraction.text import TfidfVectorizer
def extract_tfidf_features(corpus):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix.toarray()

