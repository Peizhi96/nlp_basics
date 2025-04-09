from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import euclidean_distances
import numpy as np

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




# Use Ueclidean distances to calculate cosine similarity
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
    
    # Calculate euclidean distances
    euclidean_sim = euclidean_distances(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    # Convert euclidean distances to cosine similarity
    cosine_sim = 1 / (1 + euclidean_sim)
    
    return cosine_sim[0][0]

# Write the math function for enuclidean distances
def euclidean_distance(vec1, vec2):
    """
    Calculate the Euclidean distance between two vectors.
    
    Parameters:
    vec1 (array-like): The first vector.
    vec2 (array-like): The second vector.
    
    Returns:
    float: Euclidean distance between the two vectors.
    """
    return np.sqrt(np.sum((vec1 - vec2) ** 2))
# Write the math function for cosine similarity
def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.
    
    Parameters:
    vec1 (array-like): The first vector.
    vec2 (array-like): The second vector.
    
    Returns:
    float: Cosine similarity between the two vectors.
    """
    # np.dot() calculates the dot product of the two vectors
    # The dot product is the sum of the products of the corresponding entries of the two sequences of numbers
    # For example, if vec1 = [1, 2, 3] and vec2 = [4, 5, 6], then the dot product is 1*4 + 2*5 + 3*6 = 32
    # The dot product is a measure of the similarity between two vectors
    dot_product = np.dot(vec1, vec2)
    
    # np.linalg.norm() calculates the magnitude of the vector
    # The magnitude of a vector is the square root of the sum of the squares of its components
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    
    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0.0
    
    return dot_product / (magnitude_vec1 * magnitude_vec2)


# Write the math formula for cosine similarity
# Cosine Similarity = (A . B) / (||A|| * ||B||)
# where A and B are the two vectors, ||A|| and ||B|| are the magnitudes of the vectors
# and A . B is the dot product of the vectors.
# The dot product is calculated as the sum of the products of the corresponding entries of the two sequences of numbers.
# The magnitude of a vector is calculated as the square root of the sum of the squares of its components.
# The cosine similarity is a measure of similarity between two non-zero vectors of an inner product space.  
# It is defined as the cosine of the angle between them.

# Write the math formula for euclidean distances
# Euclidean Distance = sqrt(sum((x_i - y_i)^2))
# where x_i and y_i are the components of the two vectors.
# The Euclidean distance is the length of the shortest path between two points in Euclidean space.  
# It is a measure of the straight-line distance between two points in Euclidean space.


# using dynamic programming to calculate the minimum edit distance
def edit_distance(str1, str2):
    """
    Calculate the minimum edit distance between two strings.
    
    Parameters:
    str1 (str): The first string.
    str2 (str): The second string.
    
    Returns:
    int: Minimum edit distance between the two strings.
    """
    m = len(str1)
    n = len(str2)
    
    # dp[i][j] will hold the edit distance between str1[0..i-1] and str2[0..j-1]
    # Initialize the table with size (m+1) x (n+1)
    # dp[i][0] = i (deleting all characters from str1)
    # dp[0][j] = j (inserting all characters to str1)
    
    # Create a table to store results of subproblems
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill the table in bottom-up manner
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j  # If first string is empty
            elif j == 0:
                dp[i][j] = i  # If second string is empty
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No operation needed
            else:
                dp[i][j] = min(dp[i - 1][j],     # Deletion
                               dp[i][j - 1],     # Insertion
                               dp[i - 1][j - 1]) + 1  # Substitution
    
    return dp[m][n]