#!/usr/bin/env python
"""
Character-Level Name Generation with Positional Embeddings and MLP

This script implements a simple multi-layer perceptron (MLP) to generate names.
It trains the model on a dataset of names using positional embeddings and a neural network
to predict characters step by step.

As soon as the app is opened, you will be prompted to provide a block size, the embedding dimension,
the number of neurons in the hidden layer, and a chunk size.
Then an informatory printout is shown for the first word in the dataset.
During training, a progress bar will be displayed.
After training is finished, you will be asked for the number of names to generate.
If any of the inputs (block size, embedding dimension, number of neurons, or chunk size)
are not integers, the app exits with a goodbye message.
"""

import random
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm  # for progress bar

# prompt user for block size
block_size_input = input("Enter context size (number of previous characters to consider): ")
try:
    block_size = int(block_size_input)
except ValueError:
    print("Block size must be an integer. Goodbye!")
    sys.exit()

# prompt user for embedding dimension
embedding_dim_input = input("Enter embedding size (dimensionality of character representations): ")
try:
    embedding_dim = int(embedding_dim_input)
except ValueError:
    print("Embedding dimension must be an integer. Goodbye!")
    sys.exit()

# prompt user for number of neurons in hidden layer
neurons_input = input("Enter hidden layer size (number of neurons in the MLP): ")
try:
    neurons = int(neurons_input)
except ValueError:
    print("Number of neurons must be an integer. Goodbye!")
    sys.exit()

# prompt user for chunk size (mini batch size)
chunk_size_input = input("Enter batch size (how many training samples per update): ")
try:
    chunk_size = int(chunk_size_input)
except ValueError:
    print("Chunk size must be an integer. Goodbye!")
    sys.exit()

# load names from file
with open('data/names.txt', 'r') as f:
    names = f.read().splitlines()
print(f"Saved {len(names)} names.")

# Each character is assigned a unique index, allowing us to map between characters and numbers.
# We also extend this set with special tokens for handling context windows.
chrs = list({ch for name in names for ch in name})
chrs_pos_enc = chrs + ['.' for _ in range(block_size + 1)]
i_to_s = {i: chr for i, chr in enumerate(sorted(chrs_pos_enc))}
s_to_i = {v: k for k, v in i_to_s.items()}  # a bit hacky. len is 27.

# Informatory printout of the first word (demonstration of dataset creation)
X, Y = [], []
print("Your input is being converted into training data! Below, you see how names are broken down into sequences for the model.")
print("Each step shows the sliding window of context characters ('.' represents padding) and the next character to be predicted.\n")

for name in names[:1]:
    print(name)
    name = f"{name}." 
    context = [v for v in range(0, block_size)]  # Positional encoding for each start character
    for ch in name:
        print(f"{''.join([i_to_s[w] for w in context])} --> {ch}")
        X.append(context)
        Y.append(s_to_i[ch])
        print(f"appended to datasets: {context} --> {s_to_i[ch]}")
        context = context[1:] + [s_to_i[ch]]
        print('--------')

# Instead of zero-padding, we use positional indices to provide unique context information to the model.
def build_posemb_dataset(names):
    ''' Builds a dataset with positional embedding for start characters. yay!'''
    X, Y = [], []
    for name in names:
        name = f"{name}."  # Append a stop token
        context = [v for v in range(0, block_size)]  # Positional encoding for each start character
        for ch in name:
            X.append(context)
            Y.append(s_to_i[ch])
            context = context[1:] + [s_to_i[ch]]  # Shift context window
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

# splitting data into train, validation, and test sets
n1 = int(len(names) * 0.8)
n2 = int(len(names) * 0.9)
random.shuffle(names)
Xtr, Ytr = build_posemb_dataset(names[:n1])
Xval, Yval = build_posemb_dataset(names[n1:n2])
Xte, Yte = build_posemb_dataset(names[n2:])

# Creating an Embedding Table
feature_len = Xtr.unique().shape[0] + 1
C = torch.rand((feature_len, embedding_dim))  # embedding_dim-dimensional embeddings

# Defining the Neural Network (MLP)
squashed_dims = embedding_dim * block_size
W1 = torch.rand((squashed_dims, neurons))  # neurons in the hidden layer
b1 = torch.rand(neurons)
W2 = torch.rand((neurons, feature_len))
b2 = torch.rand(feature_len)

params = [C, W1, b1, W2, b2]
loss_i = []
step_i = []

for p in params:
    p.requires_grad = True

print(f"{sum(p.nelement() for p in params)} parameters in total.")

# Training Loop
for i in tqdm(range(50000), desc="Training"):
    # forward pass
    batch = torch.randint(0, Xtr.shape[0], (chunk_size,))  # sample batch indices
    batch_emb = C[Xtr[batch]].view(chunk_size, -1)  # get embeddings
    acts = torch.sigmoid(batch_emb @ W1 + b1)  # hidden layer with sigmoid activation
    logits = acts @ W2 + b2  # output layer
    loss = F.cross_entropy(logits, Ytr[batch])  # calculate loss

    # backward pass
    for p in params:
        p.grad = None   # zeroing the gradients
    loss.backward()     # backpropagation

    # optimization
    learning_rate = 0.01 if i > 20000 else 0.1

    for p in params:
        p.data += -learning_rate * p.grad  # surfing the gradient waves

    # track steps
    loss_i.append(loss.item())
    step_i.append(i)

# evaluate the model on a separate validation set to observe generalization
val_emb = C[[Xval]].view(-1, squashed_dims)  # get embeddings for validation set
acts = torch.sigmoid(val_emb @ W1 + b1)
logits = acts @ W2 + b2
valid_loss = F.cross_entropy(logits, Yval)
print(f"trained for {len(step_i)} steps.\ntrain loss: {loss.item()}\nvalid loss: {valid_loss.item()}")

# Evaluate the model's performance on a test set.
test_emb = C[[Xte]].view(-1, squashed_dims)  # get embeddings for test set
acts = torch.sigmoid(test_emb @ W1 + b1)
logits = acts @ W2 + b2
test_loss = F.cross_entropy(logits, Yte)
print(f"Test loss: {test_loss.item()}")

# When training is finished, allow the user to enter the number of names to be generated.
while True:
    n_gen_input = input("How many names should be generated? (Enter a number, or anything else to exit): ")
    try:
        n_gen = int(n_gen_input)
    except ValueError:
        print("Goodbye!")
        sys.exit()

    # Now that training is complete, we sample new names from the model.
    for i in range(n_gen):
        context = [v for v in range(0, block_size)]  # initialize context with position indices
        word = ''
        while True:
            test_emb = C[context].view(-1, squashed_dims)  # get embeddings
            acts = torch.sigmoid(test_emb @ W1 + b1)
            logits = acts @ W2 + b2
            probs = F.softmax(logits, 1)
            pred = torch.multinomial(probs, 1).item()  # sample from probability distribution
            context = context[1:] + [pred]  # shift context window
            if i_to_s[context[-1]] == '.':  # stop if end token is predicted
                break
            word += i_to_s[context[-1]]  # append predicted character
        print(word)
