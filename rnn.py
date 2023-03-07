import torch
from torch import nn
import numpy as np

######
# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
######
is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
    
def save_model(model, hidden_dim, n_layers):
    torch.save({
                'num_hidden': n_layers,
                'num_cells': hidden_dim,
                'device': device,
                'state_dict': model.state_dict()}, 'rnn_model.pth')
    
def load_model(dict_size):
    # newmodel = Model(input_size, output_size, hidden_dim, n_layers)
    # newmodel.load_state_dict(torch.load('rnn_model.pth'))
    model = torch.load('rnn_model.pth')
    n_layers = model['num_hidden']
    hidden_dim = model['num_cells']
    device = model['device']
    seq1 = Model(input_size=dict_size, output_size=dict_size, hidden_dim=hidden_dim, n_layers=n_layers)
    seq1.load_state_dict(model['state_dict'])
    seq1.to(device)
    seq1.eval()
    return seq1  

def predict(model, character, char2int, dict_size,int2char):
    # One-hot encoding our input to fit into the model
    character = np.array([[char2int[c] for c in character]])
    character = one_hot_encode(character, dict_size, character.shape[1], 1)
    character = torch.from_numpy(character)
    character = character.to(device)
    
    out, hidden = model(character)

    prob = nn.functional.softmax(out[-1], dim=0).data
    # Taking the class with the highest probability score from the output
    char_ind = torch.max(prob, dim=0)[1].item()

    return int2char[char_ind], hidden

def sample(model, out_len, char2int, dict_size, int2char, start='hey'):
    model.eval() # eval mode
    start = start.lower()
    # First off, run through the starting characters
    chars = [ch for ch in start]
    size = out_len - len(chars)
    # Now pass in the previous characters and get a new one
    for ii in range(size):
        char, h = predict(model, chars, char2int, dict_size,int2char)
        chars.append(char)

    return ''.join(chars)
    
######
# one_hot_encode
######
def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)
    
    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u, sequence[i][u]] = 1
    return features

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        #Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out, hidden
        
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden
    
    # ref:   
    # https://github.com/gabrielloye/RNN-walkthrough/blob/master/main.ipynb?ref=floydhub-blog
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9182390