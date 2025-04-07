import nltk
from nltk.stem.porter import PorterStemmer

def porter_stem(word):
	stemmer = PorterStemmer()
	return stemmer.stem(word)