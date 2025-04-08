import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

nltk.download('stopwords')

def remove_punc_and_stop(text):
	#create a set of english stopwords
	stop_words = set(stopwords.words('english'))
	#delete punctuation
	translator = str.maketrans('', '', string.punctuation)
	text = text.translate(translator)
	#split the text into list
	words = text.split()
	filtered_words = [word for word in words if word.lower() not in stop_words ]
	return filtered_words


"""
def remove_punc_and_stop(text):
	#create a set of english stopwords
	stop_words = set(stopwords.words('english'))
	#delete punctuation
	translator = str.maketrans('', '', string.punctuation)
	text = text.translate(translator)
	#split the text into list
	words = text.split()
	stemmer = PorterStemmer()
	filtered_words = [stemmer.stem(word) for word in words if word.lower() not in stop_words ]
	return filtered_words

"""