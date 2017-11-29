import nltk
import numpy as np
from hmmlearn import hmm
import warnings
from textstat.textstat import textstat

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

grammar = ["'", "'s", "(", ")", ",", ";", ".", ":", "?", "!"]
words = [x for x in words if x not in grammar]

# Create a character -> integer mapping 
word_to_int = {word:i for i, word in enumerate(words)}
num_words = len(word_to_int)

# Create a integer -> character mapping
int_to_word = {i:word for i, word in enumerate(words)}

# Create training set for the HMM 
train_x = []
lengths = []

for idx in range(len(text)):
	if len(text[idx]) > 25:
		new_seq = nltk.word_tokenize(text[idx])
		new_seq = [int(word_to_int[word]) for word in new_seq if (word not in grammar)]

		train_x = np.concatenate([train_x, new_seq])
		lengths.append(len(new_seq))

#train_x = np.reshape(train_x, (len(train_x), 1))

# Train HMM 
print("Training HMM...")
num_states = 30
num_iter = 40
model = hmm.MultinomialHMM(n_components=num_states, n_iter=num_iter, verbose=True)
train_x = np.atleast_2d(train_x).T
train_x = train_x.astype(int)
model.fit(train_x, lengths)

A = model.transmat_
O = model.emissionprob_
A_start = model.startprob_

num_lines_tosample = 14

poem = []

for line_idx in range(num_lines_tosample):

	line = ''
	num_syllables = 0
	start_flag = True
	end_flag = False

	while True:
		# Get current syllable count
		num_syllables = np.round(textstat.syllable_count(line))

		if start_flag: 
			cur_state = np.random.choice(num_states, 1, p=A_start)[0]
			start_flag = False
		elif not end_flag:
			# Transition to new state
		    cur_state = np.random.choice(num_states, 1, p=A[cur_state, :])[0]

		# Sample word from the current state 
		sample_idx = np.random.choice(num_words, 1, p=O[cur_state, :])[0]
		sample_word = int_to_word[sample_idx]
		if sample_word == 'i':
			sample_word = 'I'

		# Syllable count of sampled word
		cur_syllable = np.round(textstat.syllable_count(sample_word))

		if cur_syllable + num_syllables == 10: 
			line += str(sample_word)
			poem.append(line)
			break
		if num_syllables == 10: 
			poem.append(line)
			break
		elif cur_syllable + num_syllables < 10:
		    # Append word to line 
		    line += str(sample_word) + " "
		else:
			end_flag = True

# Print poem 
for line_idx in range(num_lines_tosample):
	print(poem[line_idx][0].upper() + poem[line_idx][1:])








