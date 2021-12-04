import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

#Load JSON file intents.json
with open('intents.json', 'r') as f:
    intents = json.load(f)

# print(intents)
all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        #Tokenize each pattern sentence
        w = tokenize(pattern)
        all_words.extend(w) #we want to add tokenised word to all_words list and do not want to add list of tokenised word, hence use extend instead of append
        xy.append((w, tag)) #append (pattern, tag) in xy 

# print(all_words)

#ignore punctuation characters
ignore_words = ['?', '!', ',', '.'] 

#Stemmming of each word, and ignore punctuation marks
all_words = [stem(w) for w in all_words if w not in ignore_words]

#Sort all_words and store only distinct words
all_words = sorted(set(all_words))
#print(all_words)

#Sort tags and store only distinct tags
tags = sorted(set(tags))
#print(tags)

#Create training data
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label) #CrossEntropy loss - it doesnt need label to be one hot encoded, but only  class labels are required

X_train = np.array(X_train)
y_train = np.array(y_train)

#Dataset for Chatbot
class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # dataset[i] can be used to get i-th sample - Return ith item from dataset
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # Return size or number of samples
    def __len__(self):
        return self.n_samples


num_epochs = 2000
learning_rate = 0.001
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
# print(input_size, output_size)
# print(output_size, tags)

dataset = ChatDataset()
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
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Training complete. File saved to {FILE}')
