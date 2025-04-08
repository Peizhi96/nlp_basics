from gensim.models import Word2Vec

def train_word2vec_model(sentences, vector_size=100, window=5, min_count=1, workers=2):
    sentences = [sentence.split() for sentence in sentences]
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    return model

"""
optimize the hyperparameters of the Word2Vec model

param_grid = {
    'vector_size': [50, 100, 150],
    'window': [2, 5, 10],
    'min_count': [1, 2, 3],
    'sg': [0, 1],  # 0 for CBOW, 1 for Skip-gram
    'negative': [5, 10, 15],  # Number of negative samples
    'epochs': [10, 15, 20]  # Number of epochs
}

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


pipeline = Pipeline([
    ('word2vec', Word2Vec()),
    ('clf', LogisticRegression())
])

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best parameters:", best_params)
print("Best score:", best_score)

"""