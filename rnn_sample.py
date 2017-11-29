import os
import sys
import numpy as np
import random
from keras.models import Sequential, load_model
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

# Load trained LSTM network 
model = load_model('my_rnn.h5')

seed = 'from fairest creatur'

temperature = 0.5
sample_text = generate(temperature=temperature, seed=seed)

print(sample_text)


