import nltk
import numpy as np
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    word = word.lower()
    return stemmer.stem(word)

def bag_of_words(tokenised_sentence, all_words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    #Stem each word in tokenised_sentence
    sentence_words = [stem(w) for w in tokenised_sentence]

    #Initialize bag with all zeroes
    bag = np.zeros(len(all_words), dtype=np.float32)

    #Enumerate is used, as it keeps track of word and index of the word in all_words array
    for idx, w in enumerate(all_words):
        if w in sentence_words:
            bag[idx] = 1

    return bag


sentence = ["hello", "how", "are", "you"]
words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
bag = bag_of_words(sentence, words)
#print(bag)
#Testing all the above functions
# a = "Hi there, what can I do for you?"
# print(a)
# a = tokenize(a)
# print(a)

# words = ['Organize', 'organizing', 'organize']
# stemmed_words = [stem(w) for w in words]
# print(stemmed_words)
