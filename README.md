# Generating Shakesperean Literature with Hidden Markov Models and LSTM networks
Tensorflow and hmmlearn implementation of HMMs and LSTM networks trained on Shakespeare. 

Report available [here](https://github.com/khkim1/hmm-sonnet/blob/master/report/report.pdf)

## Requirements

- Python 2.7 or Python 3.3+
- [hmmlearn](https://github.com/hmmlearn/hmmlearn)
- [TensorFlow 1.0+](https://github.com/tensorflow/tensorflow/tree/r1.3)

## Usage

First, clone repository with: 

    $ git clone https://github.com/khkim1/hmm-sonnet

To train an HMM using our original implementation and sample text: 

    $ python train_hmm.py

To train an HMM using hmmlearn and sample text: 

    $ python train_hmmlearn.py

To train a LSTM network: 

    $ python train_rnn.py
