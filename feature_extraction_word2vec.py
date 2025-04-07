from gensim.models import Word2Vec

def train_word2vec_model(sentences, vector_size=100, window=5, min_count=1, workers=2):
    sentences = [sentence.split() for sentence in sentences]
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    return model