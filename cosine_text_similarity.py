from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import euclidean_distances

def cosine_text_similarity(text1, text2):
    """
    Calculate the cosine similarity between two texts.
    
    Parameters:
    text1 (str): The first text.
    text2 (str): The second text.
    
    Returns:
    float: Cosine similarity score between 0 and 1.
    """
    # Create a TfidfVectorizer instance
    vectorizer = TfidfVectorizer()
    
    # Fit and transform the texts
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]) 
    # the shape of tfidf_matrix is (2, n_features)
    # tfidf_matrix[0:1] is the first text and tfidf_matrix[1:2] is the second text
    
    # Alternatively, you can use euclidean distances
    # euclidean_sim = euclidean_distances(tfidf_matrix[0:1], tfidf_matrix[1:2])
    # cosine_sim = 1 / (1 + euclidean_sim)
    
    return cosine_sim[0][0]

# Example usage

text1 = "This is a sample text."
text2 = "This is another sample text."

similarity_score = cosine_text_similarity(text1, text2)
print(f"Cosine Similarity: {similarity_score:.4f}")