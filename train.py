import nltk
import numpy as np
import warnings
from HMM import unsupervised_HMM

warnings.simplefilter("ignore", DeprecationWarning)

# Load the shakespeare text 
text = open('data/shakespeare.txt', 'r').read()
text = text.lower()

# Split the text line by line
text = text.split('\n')
word_list = []

for idx in range(len(text)):
	if len(text[idx]) > 25:
		word_list += nltk.word_tokenize(text[idx])

words = list(set(word_list))
num_words = len(words)

grammar = ["'", "'s", "(", ")", ",", ";", ".", ":", "?"]
words = [x for x in words if x not in grammar]

# Create a character -> integer mapping 
word_to_int = {word:i for i, word in enumerate(words)}

# Create a integer -> character mapping
int_to_word = {i:word for i, word in enumerate(words)}

# Create training set for the HMM 
train_x = []

for idx in range(len(text)):
	if len(text[idx]) > 25:
		new_seq = nltk.word_tokenize(text[idx])
		new_seq = [int(word_to_int[word]) for word in new_seq if (word not in grammar)]
		train_x.append(new_seq)

# Train HMM 
print("Training HMM...")
n_states = 5
HMM = unsupervised_HMM(train_x, n_states)








