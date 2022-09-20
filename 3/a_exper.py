import torch
import torch.nn as nn
import numpy as np

class LongShortTermMemoryModel(nn.Module):

    def __init__(self, encoding_size):
        super(LongShortTermMemoryModel, self).__init__()
        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, encoding_size)  # 128 is the state size

    def reset(self):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, 1, 128)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

def create_data_encoding(index_to_char):
    n = len(index_to_char)
    encodings = torch.zeros((n, n))
    
    for i in range(n):
        encodings[i, i] = 1.0
        
    return encodings

index_to_char = [' ', 'h', 'e', 'l', 'o']

char_encodings = create_data_encoding(index_to_char)
encoding_size = len(char_encodings)

x_train = torch.stack([
    char_encodings[0],
    char_encodings[1],
    char_encodings[2],
    char_encodings[3], 
    char_encodings[3], 
    char_encodings[4], 
])[:,None,:]  # ' hello'

y_train = torch.stack([
    char_encodings[1],
    char_encodings[2],
    char_encodings[3], 
    char_encodings[3], 
    char_encodings[4], 
    char_encodings[0]
])  # 'hello '

model = LongShortTermMemoryModel(encoding_size)
optimizer = torch.optim.RMSprop(model.parameters(), 0.001)
for epoch in range(500):
    model.reset()
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 9:
        # Generate characters from the initial characters ' h'
        model.reset()
        text = ' h'
        model.f(char_encodings[0:1,None,:])
        y = model.f(char_encodings[1:2,None,:])
        text += index_to_char[y.argmax(1)]
        for c in range(50):
            arg_max = y.argmax(1)
            y = model.f(char_encodings[arg_max:arg_max+1,None,:])
            
            arg_max = y.argmax(1)
            y = model.f(char_encodings[arg_max:arg_max+1,None,:])
            text += index_to_char[y.argmax(1)]
        print(text)
