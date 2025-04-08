from sklearn.feature_extraction.text import TfidfVectorizer
def extract_tfidf_features(corpus):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix.toarray()

"""
how to optimize TfidfVectorizer
param_grid = {
    'tfidf__max_features': [10000, 15000, 20000],
    'tfidf__min_df': [2, 3, 4, 5],
    'tfidf__max_df': [0.7, 0.8, 0.9],
    'tfidf__ngram_range': [(1, 2), (1, 3)]
}
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best parameters:", best_params)
"""