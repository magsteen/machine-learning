import torch
import torch.nn as nn
from tqdm import tqdm
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
    encodings = [[0.0 for col in range(n)] for row in range(n)]
        
    for i in range(n):
        encodings[i][i] = 1.0
        
    return encodings

index_to_char = [' ', 'a', 'c', 'f', 'h', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't'] # len 13 -> last index 12

char_encodings = create_data_encoding(index_to_char)
encoding_size = len(char_encodings)

hat = torch.tensor([
    [char_encodings[4]],
    [char_encodings[1]],
    [char_encodings[12]],
    [char_encodings[0]],
])
rat = torch.tensor([
    [char_encodings[10]],
    [char_encodings[1]],
    [char_encodings[12]],
    [char_encodings[0]],
])
cat = torch.tensor([
    [char_encodings[2]],
    [char_encodings[1]],
    [char_encodings[12]],
    [char_encodings[0]],
])
flat = torch.tensor([
    [char_encodings[4]],
    [char_encodings[5]],
    [char_encodings[1]],
    [char_encodings[12]]
])
matt = torch.tensor([
    [char_encodings[6]], 
    [char_encodings[1]], 
    [char_encodings[12]], 
    [char_encodings[12]],
])
cap = torch.tensor([
    [char_encodings[2]], 
    [char_encodings[1]], 
    [char_encodings[9]], 
    [char_encodings[0]],
])
son = torch.tensor([
    [char_encodings[11]], 
    [char_encodings[8]], 
    [char_encodings[7]], 
    [char_encodings[0]],
])

x_train = torch.stack([
    hat,
    rat,
    cat,
    flat,
    matt,
    cap,
    son
])#[:,:,None,:]

# y_train = torch.tensor([
#     '🎩',
#     '🐀',
#     '🐈',
#     '🏢',
#     '🧔',
#     '🧢',
#     '👶'
# ])

# y_train = torch.tensor([
#     [0.0],
#     [1.0],
#     [2.0],
#     [3.0],
#     [4.0],
#     [5.0],
#     [6.0]
# ])

test = np.eye(7)

y_train = torch.tensor([
    [test[0],test[0],test[0],test[0]],
    [test[1],test[1],test[1],test[1]],
    [test[2],test[2],test[2],test[2]],
    [test[3],test[3],test[3],test[3]],
    [test[4],test[4],test[4],test[4]],
    [test[5],test[5],test[5],test[5]],
    [test[6],test[6],test[6],test[6]]
])

emojis = {
    0: '🎩',
    1: '🐀',
    2: '🐈',
    3: '🏢',
    4: '🧔',
    5: '🧢',
    6: '👶',
}

print(x_train.size())
print(y_train.size())

model = LongShortTermMemoryModel(encoding_size)
optimizer = torch.optim.RMSprop(model.parameters(), 0.001)
for epoch in tqdm(range(50)):
    for i in range(x_train.size()[0]):
        model.reset()
        model.loss(x_train[i], y_train[i]).backward()
        optimizer.step()
        optimizer.zero_grad()

# for i in range(x_train.size()[0]):
#     model.f(x_train[i])

def generate_emoji(string):
    y = -1
    model.reset()
    for i in range(len(string)):
        char_index = index_to_char.index(string[i])
        y = model.f(torch.tensor([[char_encodings[char_index]]], dtype=torch.float))
    print(emojis[y.argmax(1).item()])

generate_emoji('rt')
generate_emoji('rats')
generate_emoji('m')
generate_emoji('fl')
generate_emoji('rp')
generate_emoji('son')
