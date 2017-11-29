import nltk
import numpy as np
from hmmlearn import hmm
import warnings
from textstat.textstat import textstat
import train_hmm_rhyme

warnings.simplefilter("ignore", DeprecationWarning)

# Load the shakespeare sonnets
with open('data/shakespeare_updated.txt') as infile:
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

# Generate the poem, and track emission matrices for analysis
poem = ''
print("Training HMM on leading quatrains...")
hmm_lead_results = train_hmm_rhyme.generate(leading_quatrains, 8, 'start')
poem += hmm_lead_results[0]
O_start_quats = hmm_lead_results[1]
print("Training HMM on voltas...")
hmm_volta_results = train_hmm_rhyme.generate(voltas, 4, 'volta')
poem += hmm_volta_results[0]
O_voltas = hmm_volta_results[1]
print("Training HMM on couplets...")
hmm_couplet_results = train_hmm_rhyme.generate(couplets, 2, 'couplet')
poem += hmm_couplet_results[0]
O_couplets = hmm_couplet_results[1]

print(poem)
