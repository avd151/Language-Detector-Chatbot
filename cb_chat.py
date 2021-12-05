import random
import json
import torch
from cb_model import NeuralNet
from cb_nltk_utils import BagOfWords
from ld_language_detector import LanguageModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bow = BagOfWords()
lm = LanguageModel()

def generate_response(sentence:str) -> str:
    #Detect language
    if ("Detect: " in sentence) or ("detect: " in sentence):
        language =  lm.predict(sentence[8: ]) #detect:  at the start
        return f"The language detected is {language}."
    elif ("Identify: " in sentence) or ("Identify: " in sentence):
        language = lm.predict(sentence[8:])  # identify:  at the start
        return f"The language detected is {language}."
    else:
        #General Chatbot
        sentence = bow.tokenize(sentence)
        X = bow.bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.1:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    return f"{random.choice(intent['responses'])}"
        else:
            return f"I do not understand it.. Please use available options to detect/identify language"


if __name__ == '__main__':
    print("Let's Chat!")
    print("Type detect: <text> or identify: <text> to identify language from the text")
    print("Type quit to exit")
    while True:
        sentence = input("You: ")
        if (sentence == "quit") or (sentence == "Quit") or (sentence == "exit") or (sentence == "Exit"):
            break
        else:
            print(generate_response(sentence))
    