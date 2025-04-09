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
    
"""
implement HMM for POS tagging using viterbi algorithm
import numpy as np

def viterbi(obs, states, start_p, trans_p, emit_p):
    # obs: observed sequence, e.g., ['the', 'dog', 'barks']
    # states: all possible states, e.g., ['NOUN', 'VERB', 'ADJ']
    # start_p: initial probabilities, e.g., {state: probability}
    # trans_p: transition probabilities, e.g., {from_state_1: {to_state2_1: probability}}
    # emit_p: emission probabilities, e.g., {state: {word: probability}}
    
    # V is a list of dictionaries, where each dictionary corresponds to a time step
    # and contains the maximum probability for each state at that time step
    # path is a dictionary that stores the best path to each state
    # at the current time step
    V = [{}]
    path = {}

    
    # initialize the first time step
    for state in states:
        # for each state, calculate the probability of starting in that state
        # initial probability * emission probability
        V[0][state] = start_p[state] * emit_p[state][obs[0]]
        # initialize the path for each state
        path[state] = [state]
    
    # iterate throught the next time steps, from 1 to T-1
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}
        
        # interate throught all possible forward state y0
        # forward probability * transition probability * emission probability
        # maximize is to get y0 that gives the maximum probability
        for y in states:
            (prob, state) = max(
                (V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) 
                for y0 in states
            )
            # update the maximum probability for the current state
            V[t][y] = prob
            # update the path to the current state
            newpath[y] = path[state] + [y]
        # replace the old path with the new path
        path = newpath
    
    # find the state with the highest probability at the last time step
    # and return the maximum probability and the best path
    (prob, state) = max((V[-1][y], y) for y in states)
    return (prob, path[state])
"""