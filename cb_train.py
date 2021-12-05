import json
from cb_nltk_utils import BagOfWords
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from cb_model import NeuralNet

class Train(object):
    def __init__(self) -> None:
        super().__init__()
        self.bow = BagOfWords()
        self.initData()
        self.loadIntents()
        self.tokanize()
        self.train()

    def initData(self):
        self.all_words = []
        self.tags = []
        self.xy = []
        self.ignore_words = ['?', '!', ',', '.'] 

    #Load JSON file intents.json
    def loadIntents(self):
        with open('intents.json', 'r') as f:
            self.intents = json.load(f)

    def tokanize(self):
        for intent in self.intents['intents']:
            tag = intent['tag']
            self.tags.append(tag)
            for pattern in intent['patterns']:
                #Tokenize each pattern sentence
                w = self.bow.tokenize(pattern)
                self.all_words.extend(w) #we want to add tokenised word to all_words list and do not want to add list of tokenised word, hence use extend instead of append
                self.xy.append((w, tag)) #append (pattern, tag) in xy 

    def stemWord(self):
        #Stemmming of each word, and ignore punctuation marks
        self.all_words = [self.bow.stem(w) for w in self.all_words if w not in self.ignore_words]

        #Sort all_words and store only distinct words
        self.all_words = sorted(set(self.all_words))

        #Sort tags and store only distinct tags
        self.tags = sorted(set(self.tags))

    def train(self):
        #Create training data
        self.x_train = []
        self.y_train = []

        for (pattern_sentence, tag) in self.xy:
            bag = self.bow.bag_of_words(pattern_sentence, self.all_words)
            self.x_train.append(bag)

            label = self.tags.index(tag)
            self.y_train.append(label) #CrossEntropy loss - it doesnt need label to be one hot encoded, but only  class labels are required

        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)

#Dataset for Chatbot
class ChatDataset(Dataset):

    def __init__(self):
        self.t = Train()
        self.n_samples = len(self.t.x_train)
        self.x_data = self.t.x_train
        self.y_data = self.t.y_train

    # dataset[i] can be used to get i-th sample - Return ith item from dataset
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # Return size or number of samples
    def __len__(self):
        return self.n_samples

#train
if __name__ == '__main__':
    dataset = ChatDataset()
    num_epochs = 2000
    learning_rate = 0.001
    batch_size = 8
    hidden_size = 8
    output_size = len(dataset.t.tags)
    input_size = len(dataset.t.x_train[0])

    train_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0) #num_workers - used in multiprocessing using GPU

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # Train the model
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)

            # Forward pass
            outputs = model(words)
            # if y would be one-hot, we must apply
            # labels = torch.max(labels, 1)[1]
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print(f'final loss: {loss.item():.4f}')

    #Save data
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": dataset.t.all_words,
        "tags": dataset.t.tags
    }

    FILE = "data.pth"
    torch.save(data, FILE)

    print(f'Training complete. File saved to {FILE}')