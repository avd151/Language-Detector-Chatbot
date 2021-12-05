import pandas as pd
import re
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import pickle

warnings.simplefilter("ignore")

class LanguageModel(object):
    def __init__(self) -> None:
        super().__init__()

        # Loading the dataset
        self.data = pd.read_csv("./Data-Set/Language Detection.csv")
    
        # separating the independent and dependant features
        self.x = self.data["Text"]
        self.y = self.data["Language"]

        self.preprocess()

    def preprocess(self) -> None:
        # converting categorical variables to numerical
        self.le = LabelEncoder()
        self.y = self.le.fit_transform(self.y)

        # creating a list for appending the preprocessed text
        self.data_list = []
        # iterating through all the text
        for text in self.x:
            # removing the symbols and numbers
            text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
            text = re.sub(r'[[]]', ' ', text)
            # converting the text to lower case
            text = text.lower()
            # appending to data_list
            self.data_list.append(text)

    def createBagOfWords(self)-> None:
        # creating bag of words using countvectorizer
        self.cv = CountVectorizer()
        X = self.cv.fit_transform(self.data_list).toarray()
        pickle.dump(self.cv, open('cv.pkl', 'wb'))

    def createModel(self) -> None:
        #train test splitting
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size = 0.20)
        #model creation and prediction
        self.model = MultinomialNB()
        self.model.fit(x_train, y_train)
        pickle.dump(self.model, open('model.pkl', 'wb'))

    def train(self) -> None:
        self.createBagOfWords()
        self.createModel()

    def predict(self, text:str) -> str:
        self.cv = pickle.load(open('cv.pkl', 'rb'))
        x = self.cv.transform([text]).toarray()
        
        self.model = pickle.load(open('model.pkl', 'rb'))
        lang = self.model.predict(x)

        lang = self.le.inverse_transform(lang)
        
        return lang[0]


#testing predictions
if __name__ == '__main__':

    lm = LanguageModel()

    # English
    lan = lm.predict("Analytics Vidhya provides a community based knowledge portal for Analytics and Data Science professionals")
    print(lan)
    # French
    lan = lm.predict("Analytics Vidhya fournit un portail de connaissances basé sur la communauté pour les professionnels de l'analyse et de la science des données")
    print(lan)
    # Arabic
    lan = lm.predict("توفر Analytics Vidhya بوابة معرفية قائمة على المجتمع لمحترفي التحليلات وعلوم البيانات")
    print(lan)
    # Spanish
    lan = lm.predict("Analytics Vidhya proporciona un portal de conocimiento basado en la comunidad para profesionales de Analytics y Data Science.")
    print(lan)
    # Malayalam
    lan = lm.predict("അനലിറ്റിക്സ്, ഡാറ്റാ സയൻസ് പ്രൊഫഷണലുകൾക്കായി കമ്മ്യൂണിറ്റി അധിഷ്ഠിത വിജ്ഞാന പോർട്ടൽ അനലിറ്റിക്സ് വിദ്യ നൽകുന്നു")
    print(lan)
    # Russian
    lan = lm.predict("Analytics Vidhya - это портал знаний на базе сообщества для профессионалов в области аналитики и данных.")
    print(lan)