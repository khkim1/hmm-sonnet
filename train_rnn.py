'''
LSTMs for poetry generation
Author: John Kim 
Email: khkim@caltech.edu
'''

import os
import sys
import numpy as np
import random
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout

# Function to generate sample text 
def generate(temperature=1.0, seed=None):

    cur_seed = seed

    while len(cur_seed) < 1200:
        x = np.zeros((1, win_size, num_chars))

        for t, char in enumerate(seed):
            x[0, t, char_to_int[char]] = 1.

        # Obtain output prob distribution
        probs = model.predict(x, verbose=0)[0]

        # Temperature sampling
        a = np.log(probs)/temperature
        d = np.exp(a)/np.sum(np.exp(a))
        choices = range(len(probs))

        next_idx = np.random.choice(choices, p=d)
        next_char = int_to_char[next_idx]

        cur_seed += next_char
        seed = seed[1 : ] + next_char

    return cur_seed 

#------------------------------- Clean up data ------------------------------------

# Load the text
text = open('data/combined.txt', 'r').read()
text = text.lower()

# Create a list of the unique characters in the text
chars = list(set(text))
num_chars = len(chars)

# Create a character -> integer mapping 
char_to_int = {ch:i for i, ch in enumerate(chars)}

# Create a integer -> character mapping
int_to_char = {i:ch for i, ch in enumerate(chars)}

# Window size
win_size = 20

# Stride size 
stride = 3

# Create RNN architecture: 128 LSTM units 
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(win_size, num_chars)))
model.add(LSTM(512, return_sequences=False))
model.add(Dense(num_chars))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam') # Try rmsprop 

# Inputs to the RNN 
in_sequence = []
output = []
for i in range(0, len(text) - win_size, stride):
    # Take a chunk of the corpus with size 50
    in_sequence.append(text[i : i + win_size])

    # Character following the input sequence
    output.append(text[i + win_size])

# Create one-hot encoded inputs and outputs 
X = np.zeros((len(in_sequence), win_size, num_chars))
y = np.zeros((len(in_sequence), num_chars))

for i, example in enumerate(in_sequence):
    for t, char in enumerate(example):
        X[i, t, char_to_int[char]] = 1
    y[i, char_to_int[output[i]]] = 1


#----------------------------------- Training --------------------------------------

# Initialize training parameters 
tot_epochs = 100
batch = 128
temperature = 0.2
counter = 0

for i in range(tot_epochs):

    # Increment training loop counter
    counter += 1

    # Train the RNN for one epoch 
    model.fit(X, y, batch_size=batch, nb_epoch=1)

    # Generate sample text 
    seed_idx = random.randint(0, len(text)-win_size-1)
    seed = text[seed_idx : seed_idx+win_size]
    sample_text = generate(temperature=temperature, seed=seed)

    # Print the results
    print('Training Epoch: %d' %counter)
    print(sample_text)


