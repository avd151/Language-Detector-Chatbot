import nltk
import numpy as np
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

class BagOfWords(object):
    def __init__(self) -> None:
        super().__init__()
        self.stemmer = PorterStemmer()

    def tokenize(self, sentence):
        '''
        Break the text into tokens.
        Example:
        sentence = "Hi there, what can I do for you?"
        tokens = ['Hi', 'there', ',', 'what', 'can', 'I', 'do', 'for', 'you', '?']
        '''
        return nltk.word_tokenize(sentence)

    def stem(self, word):
        '''
        Return stem of the word.
        example:
        words = ['Organize', 'organizing', 'organize']
        stem = ['organ', 'organ', 'organ']
        '''
        word = word.lower()
        return self.stemmer.stem(word)

    def bag_of_words(self, tokenised_sentence, all_words):
        """
        return bag of words array:
        1 for each known word that exists in the sentence, 0 otherwise
        example:
        sentence = ["hello", "how", "are", "you"]
        words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
        bag   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
        """
        #Stem each word in tokenised_sentence
        sentence_words = [self.stem(w) for w in tokenised_sentence]

        #Initialize bag with all zeroes
        bag = np.zeros(len(all_words), dtype=np.float32)

        #Enumerate is used, as it keeps track of word and index of the word in all_words array
        for idx, w in enumerate(all_words):
            if w in sentence_words:
                bag[idx] = 1

        return bag


#testing all the functions
if __name__ == '__main__':
    bow = BagOfWords()

    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]

    bag = bow.bag_of_words(sentence, words)
    print(bag)

    a = "Hi there, what can I do for you?"
    print(a)
    a = bow.tokenize(a)
    print(a)

    words = ['Organize', 'organizing', 'organize']
    stemmed_words = [bow.stem(w) for w in words]
    print(stemmed_words)