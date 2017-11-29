import numpy as np
from hmmlearn import hmm
from textstat.textstat import textstat
import nltk
import curses
from curses.ascii import isdigit
import nltk
from nltk.corpus import cmudict
import pronouncing as pro
import random


# Utilities for syllable counting
d = cmudict.dict()
PUNCTUATION = ["'", "'s", "(", ")", ",", ";", ".", ":", "?", "!"]
def nsyl(word):
	''' Return the number of syllables in word.'''
	try:
		res = [len(list(y for y in x if isdigit(y[-1]))) for x in d[word.lower()]][0]
	except:
		res = np.round(textstat.syllable_count(word))
	return res

def count_syllables(sent):
	''' Return the number of syllables in a given sentence.'''
	words = sent.split()
	num_syllables = 0
	for word in words:
		num_syllables += nsyl(word)
	return num_syllables

def get_rhyme_dict(lines, subsize):
	''' Build a dictionary of rhyming words based on the given lines. subsize
	indicates how many lines come from one poem. For example, if lines consists
	of all couplets, subsize would be 2.'''
	res = {}
	i = 0
	while i < len(lines):
		curr_poem = lines[i : i+subsize]
		if subsize == 8:
			res[get_last_word(curr_poem[0])] = get_last_word(curr_poem[2])
			res[get_last_word(curr_poem[2])] = get_last_word(curr_poem[0])

			res[get_last_word(curr_poem[1])] = get_last_word(curr_poem[3])
			res[get_last_word(curr_poem[3])] = get_last_word(curr_poem[1])

			res[get_last_word(curr_poem[4])] = get_last_word(curr_poem[6])
			res[get_last_word(curr_poem[6])] = get_last_word(curr_poem[4])

			res[get_last_word(curr_poem[5])] = get_last_word(curr_poem[7])
			res[get_last_word(curr_poem[7])] = get_last_word(curr_poem[5])

		elif subsize == 4:
			res[get_last_word(curr_poem[0])] = get_last_word(curr_poem[2])
			res[get_last_word(curr_poem[2])] = get_last_word(curr_poem[0])

			res[get_last_word(curr_poem[1])] = get_last_word(curr_poem[3])
			res[get_last_word(curr_poem[3])] = get_last_word(curr_poem[1])

		elif subsize == 2:
			res[get_last_word(curr_poem[0])] = get_last_word(curr_poem[1])
			res[get_last_word(curr_poem[1])] = get_last_word(curr_poem[0])

		else:
			print("ERROR: invalid subsize in get_rhyme_dict(): " + subsize)
		i += subsize
	return res

def get_punc_prob_dict(lines):
	''' Get the probability of each line ending puncation in lines. Ignore
	periods.'''
	count = 0
	res = {end: 0 for end in [',', ';', ':', '-']}
	for line in lines:
		line_end = line[-1]
		if line_end in [',', ';', ':', '-']:
			count += 1
			res[line_end] += 1
	return res

def get_last_word(line):
	last_word = line.split()[-1]
	last_word = ''.join([c for c in last_word if c not in PUNCTUATION])
	return last_word

def generate(lines, n_lines, section_type):
	''' Takes in lines (a list of strings) and produces n_lines of poetry using
	an HMM model. Assumes extraneous lines (new lines, poem numbers) have
	already been removed. section_type specifies which portion of the sonnet
	to generate (and therefore its rhyme scheme), and takes the following
	values:
			'start' - first two quatrains of the sonnet (ABABCDCD)
			'volta' - third quatrain of the sonnet (EFEF)
			'couplet' - last two lines of the sonnet (GG)
	'''
	# Get the unique words
	word_list = [nltk.word_tokenize(line) for line in lines]
	word_list = [word for sublist in word_list for word in sublist]

	words = list(set(word_list))

	punctuation = PUNCTUATION
	words = [x for x in words if x not in punctuation]

	# Create a word -> integer mapping
	word_to_int = {word:i for i, word in enumerate(words)}
	num_words = len(word_to_int)

	# Create a integer -> word mapping
	int_to_word = {i:word for i, word in enumerate(words)}

	print("Number of unique words is: " + str(len(words)))

	# Create the training set
	train_x = []
	lengths = []

	for line in lines:
		new_seq = nltk.word_tokenize(line)
		new_seq = [int(word_to_int[word]) for word in new_seq if (word not in punctuation)]
		train_x = np.concatenate([train_x, new_seq])
		lengths.append(len(new_seq))

	num_states = 15
	num_iter = 50
	model = hmm.MultinomialHMM(n_components=num_states, n_iter=num_iter, verbose=True)
	train_x = np.atleast_2d(train_x).T
	train_x = train_x.astype(int)
	model.fit(train_x, lengths)

	A = model.transmat_
	O = model.emissionprob_
	A_start = model.startprob_

	num_lines_to_sample = n_lines

	# Get the proper rhyming dictionary
	if section_type == 'start':
		rhyme_dict = get_rhyme_dict(lines, 8)
	elif section_type == 'volta':
		rhyme_dict = get_rhyme_dict(lines, 4)
	elif section_type == 'couplet':
		rhyme_dict = get_rhyme_dict(lines, 2)
	else:
		print('ERROR: Invalid section type in generate()')
	print(rhyme_dict)

	# Line enders
	all_enders = [',', ';', ':', '-', '.', ' ']
	enders_no_period = [',', ';', ':', '-', ' ']

	poem = []
	line_cnt = 0
	while line_cnt < num_lines_to_sample:

		if section_type == 'start':
			if line_cnt == 0: # Start a new rhyme
				A_rhyme = random.choice(list(rhyme_dict.keys()))
				line = A_rhyme + random.choice(enders_no_period)
			elif line_cnt == 1:
				B_rhyme = random.choice(list(rhyme_dict.keys()))
				line = B_rhyme + random.choice(all_enders)
			elif line_cnt == 2:
				line = rhyme_dict[A_rhyme] + random.choice(all_enders)
			elif line_cnt == 3:
				line = rhyme_dict[B_rhyme] + random.choice(all_enders)
			elif line_cnt == 4:
				C_rhyme = random.choice(list(rhyme_dict.keys()))
				line = C_rhyme + random.choice(all_enders)
			elif line_cnt == 5:
				D_rhyme = random.choice(list(rhyme_dict.keys()))
				line = D_rhyme + random.choice(all_enders)
			elif line_cnt == 6:
				line = rhyme_dict[C_rhyme] + random.choice(enders_no_period)
			elif line_cnt == 7:
				line = rhyme_dict[D_rhyme] + '.'
			else:
				print("ERROR: invalid line_cnt in generate()")

		elif section_type == 'volta':
			if line_cnt == 0: # Start a new rhyme
				E_rhyme = random.choice(list(rhyme_dict.keys()))
				line = E_rhyme + random.choice(enders_no_period)
			elif line_cnt == 1:
				F_rhyme = random.choice(list(rhyme_dict.keys()))
				line = F_rhyme + random.choice(all_enders)
			elif line_cnt == 2:
				line = rhyme_dict[E_rhyme] + random.choice(enders_no_period)
			elif line_cnt == 3:
				line = rhyme_dict[F_rhyme] + '.'

		else:
			if line_cnt == 0:
				seed = random.choice(list(rhyme_dict.keys()))
				line = seed + random.choice(enders_no_period)
			else:
				line = rhyme_dict[seed] + '.'

		start_flag = True
		end_flag = False

		while True:
			# Get current syllable count
			num_syllables = count_syllables(line)

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
			new_syll_count = count_syllables(line + str(sample_word))

			if new_syll_count == 10:

				# The line is complete, so add it to the poem
				line = str(sample_word) + ' ' + line
				poem.append(line)
				line_cnt += 1
				break
			elif new_syll_count < 10:
				line = str(sample_word) + ' ' + line
			else:
				end_flag = True

	# Return final poem
	final_poem = ''
	for line_idx in range(num_lines_to_sample):
		final_poem += poem[line_idx][0].upper() + poem[line_idx][1:] + '\n'
	return final_poem, model.emissionprob_, process_emissions(O, int_to_word)

def process_emissions(O, int_to_word):
	'''A helper to make analysis of emission probablities easier.'''
	res = []
	for sublist in O:
		sorted_idxs = sorted(range(len(sublist)), key=lambda x: sublist[x])
		top_10_idxs = sorted_idxs[:10]
		top_10_words = [int_to_word[i] for i in top_10_idxs]
		res.append(top_10_words)
	return res
