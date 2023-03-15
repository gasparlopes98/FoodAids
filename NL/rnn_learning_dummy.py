import torch
from torch import nn
import rnn

######
# Adds apping to arrays so they are the same lenght
######
def padding (text,maxlen):
    for i in range(len(text)):
        while len(text[i])<maxlen:
            text[i] += ' '
    return text

text = ['hey how are you','good i am fine','have a nice day']
maxlen = len(max(text, key=len))
text=padding(text,maxlen)

# Join all the sentences together and extract the unique characters from the combined sentences
chars = set(''.join(text))

# Creating a dictionary that maps integers to the characters
int2char = dict(enumerate(chars))

# Creating another dictionary that maps characters to integers
char2int = {char: ind for ind, char in int2char.items()}

# Creating lists that will hold our input and target sequences
input_seq = []
target_seq = []

for i in range(len(text)):
    # Remove last character for input sequence
    input_seq.append(text[i][:-1])
    
    # Remove firsts character for target sequence
    target_seq.append(text[i][1:])
    print("Input Sequence: {}\nTarget Sequence: {}".format(input_seq[i], target_seq[i]))
    
# Convert input and target sequences to sequences of integers instead of characters by mapping them using the dictionaries we created above. This will allow us to one-hot-encode our input sequence subsequently.
for i in range(len(text)):
    input_seq[i] = [char2int[character] for character in input_seq[i]]
    target_seq[i] = [char2int[character] for character in target_seq[i]]

dict_size = len(char2int)
seq_len = maxlen - 1
batch_size = len(text)

input_seq = rnn.one_hot_encode(input_seq, dict_size, seq_len, batch_size)
print("Input shape: {} --> (Batch Size, Sequence Length, One-Hot Encoding Size)".format(input_seq.shape))

# move the data from numpy arrays to Torch Tensors 
input_seq = torch.from_numpy(input_seq)
target_seq = torch.Tensor(target_seq)

# Define hyperparameters
n_epochs = 100
lr = 0.01

if(input('Do you want to train the model? (y/n)')=='y'):
    # Instantiate the model with hyperparameters
    model = rnn.Model(input_size=dict_size, output_size=dict_size, hidden_dim=12, n_layers=1)
    # We'll also set the model to the device that we defined earlier (default is CPU)
    model = model.to(rnn.device)
    
    # Define Loss, Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training Run
    input_seq = input_seq.to(rnn.device)
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad() # Clears existing gradients from previous epoch
        #input_seq = input_seq.to(device)
        output, hidden = model(input_seq)
        output = output.to(rnn.device)
        target_seq = target_seq.to(rnn.device)
        loss = criterion(output, target_seq.view(-1).long())
        loss.backward() # Does backpropagation and calculates gradients
        optimizer.step() # Updates the weights accordingly
        
        if epoch%10 == 0:
            print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))
            
        rnn.save_model(model, hidden_dim=12, n_layers=1)
        # torch.save(model, 'rnn_model.pth')
else:
    model = rnn.load_model(dict_size)

# Test / Prediction
print(rnn.sample(model, 15, char2int, dict_size, int2char, 'good'))