import numpy as np

def ibm_model1(parallel_corpus, source_sentences):
    # build vocabulary for source and target languages
    # parallel_corpus is a list of tuples (source_sentence, target_sentence)
    # source_sentences is the sentence to be translated
    # Example: parallel_corpus = [("hello world", "hola mundo"), ("goodbye world", "adiós mundo")]
    # source_sentences = "hello world"
    # This function implements IBM Model 1 for word-based translation   
    source_vocab = set() # store unique words in source language
    target_vocab = set() # store unique words in target language
    
    # iterate through the parallel corpus to build vocabularies
    for source, target in parallel_corpus:
        # split sentences into words
        # and add them to the respective vocabularies
        for word in source.split():
            source_vocab.add(word)
        # split sentences into words
        # and add them to the respective vocabularies
        for word in target.split():
            target_vocab.add(word)
    
    # initialize translation probabilities uniformly
    translation_probs = {}
    
    # for each word in source language, assign equal probability to each word in target language
    for source_word in source_vocab:
        for target_word in target_vocab:
            translation_probs[(source_word, target_word)] = 1.0 / len(target_vocab)
    
    # EM algorithm for estimating translation probabilities
    num_iterations = 10
    for _ in range(num_iterations):
        # create count and total dictionaries to store counts and totals
        count = {}
        total = {}
        
        # initialize counts and totals
        for source_word in source_vocab:
            total[source_word] = 0 #initialize total counts for each source word
            for target_word in target_vocab:
                count[(source_word, target_word)] = 0 #initialize counts for each source-target pair
        
        # iterate through the parallel corpus
        for source, target in parallel_corpus:
            source_words = source.split() # split source sentence into words
            target_words = target.split() # split target sentence into words
            
            # calculate normalization factor
            # s_total is the sum of translation probabilities for each target word
            s_total = {}
            for target_word in target_words:
                s_total[target_word] = 0
                
                # for each source word, add the translation probability to the total
                for source_word in source_words:
                    s_total[target_word] += translation_probs[(source_word, target_word)]
                    
            # update counts and totals (collect statistics)
            for target_word in target_words:
                for source_word in source_words:
                    
                    # calculate delta as the ratio of translation probability to normalization factor
                    delta = translation_probs[(source_word, target_word)] / s_total[target_word]
                    
                    # update counts and totals
                    count[(source_word, target_word)] += delta
                    total[source_word] += delta
        
        # update translation probabilities
        for source_word in source_vocab:
            for target_word in target_vocab:
                if total[source_word] > 0:
                    # update translation probability as the ratio of count to total
                    translation_probs[(source_word, target_word)] = count[(source_word, target_word)] / total[source_word]
                else:
                    # if total is zero, set translation probability to zero
                    # to avoid division by zero
                    translation_probs[(source_word, target_word)] = 0.0
    
    # translate the source sentence
    # by finding the target word with the highest translation probability
    source_words = source_sentences.split()
    translation = []
    for source_word in source_words:
        max_prob = 0
        best_target_word = None
        # iterate through target vocabulary to find the best translation
        for target_word in target_vocab:
            if translation_probs[(source_word, target_word)] > max_prob:
                max_prob = translation_probs[(source_word, target_word)]
                best_target_word = target_word
        translation.append(best_target_word)
    return ' '.join(translation)

# Example usage
parallel_corpus = [
    ("hello world", "hola mundo"),
    ("goodbye world", "adiós mundo"),
    ("hello everyone", "hola a todos"),
]
source_sentences = "hello world"
translation = ibm_model1(parallel_corpus, source_sentences)
print(f"Translation of '{source_sentences}': {translation}")