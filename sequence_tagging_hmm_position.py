import nltk 
from nltk.corpus import brown
from nltk.tag import hmm
from sklearn.model_selection import train_test_split


def hmm_pos_tagging(text):
    nltk.download('brown')
    # Load the Brown corpus
    sentences = brown.tagged_sents()
    
    # Split the data into training and testing sets
    train_data, test_data = train_test_split(sentences, test_size=0.2, random_state=42)
    
    # Train the HMM POS tagger
    trainer = hmm.HiddenMarkovModelTrainer()
    
    tagger = trainer.train_supervised(train_data)
    # Test the tagger on a sample text
    test_text = text.split()
    tagged_text = tagger.tag(test_text)
    return tagged_text


if __name__ == "__main__":
    text = "The quick brown fox jumps over the lazy dog."
    tagged_text = hmm_pos_tagging(text)
    print(tagged_text)