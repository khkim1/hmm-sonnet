import nltk
import numpy as np
from hmmlearn import hmm
import warnings
from textstat.textstat import textstat
import train_hmm_rhyme

# Ignore the warnings output by hmmlearn
warnings.simplefilter("ignore", DeprecationWarning)

# Load the shakespeare sonnets
with open('data/shakespeare.txt') as infile:
	lines = infile.readlines()
lines = [line.strip().lower() for line in lines if len(line.strip()) > 25]

# Split into the first two quatrains, the third quatrain (volta), and the
# last two lines (couplet)
leading_quatrains = []
voltas = []
couplets = []
i = 0
while i < len(lines):
	leading_quatrains.extend(lines[i: i+8])
	voltas.extend(lines[i+8: i+12])
	couplets.extend(lines[i+12: i+14])
	i += 14

# Generate the poem
num_poems = 10

print("Training HMM on leading quatrains...")
poem_quatrains = train_hmm_rhyme.generate(leading_quatrains, 8, 'start', num_poems)
print("Training HMM on voltas...")
poem_voltas = train_hmm_rhyme.generate(voltas, 4, 'volta', num_poems)
print("Training HMM on couplets...")
poem_couplets = train_hmm_rhyme.generate(couplets, 2, 'couplet', num_poems)

for idx in range(num_poems):
	poem = ''
	poem += poem_quatrains[idx]
	poem += poem_voltas[idx]
	poem += poem_couplets[idx]
	print(poem)
