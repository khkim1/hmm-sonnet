import random
import numpy as np

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0.
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.
            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            D:          Number of observations.
            A:          The transition matrix.
            O:          The observation matrix.
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''
        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for i in range(self.L)]

    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    Output sequence corresponding to x with the highest
                        probability.
        '''
        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for i in range(self.L)] for j in range(M + 1)]
        seqs = [['' for i in range(self.L)] for j in range(M + 1)]

        # Initialize the first row
        for i in range(self.L):
            probs[1][i] = self.A_start[i] * self.O[i][x[0]]

        seqs[1][:] = [str(i) for i in range(self.L)]

        for ridx in range(2, M + 1):
            for cidx in range(self.L):
                max_idx = 0
                argmax_yj_a = 0
                for upper_idx in range(self.L):
                    yj_a = probs[ridx-1][upper_idx] * self.A[upper_idx][cidx] * self.O[cidx][x[ridx-1]]
                    if yj_a > argmax_yj_a:
                        max_idx = upper_idx
                        argmax_yj_a = yj_a

                probs[ridx][cidx] = argmax_yj_a
                seqs[ridx][cidx] = seqs[ridx-1][max_idx] + str(cidx)

        last_row = probs[M][:]
        max_seq = seqs[M][last_row.index(max(last_row))]
        return max_seq

    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.
                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for i in range(self.L)] for j in range(M + 1)]

        for i in range(self.L):
            alphas[1][i] = self.O[i][x[0]] * self.A_start[i]

        for ridx in range(2, M + 1):
            for cidx in range(self.L):
                upper_sum = 0
                for upper_idx in range(self.L):
                    upper_sum += alphas[ridx-1][upper_idx]*self.A[upper_idx][cidx]

                alphas[ridx][cidx] = upper_sum*self.O[cidx][x[ridx - 1]]

        return alphas

    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.
                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.
        '''
        M = len(x)      # Length of sequence.
        betas = [[0. for i in range(self.L)] for j in range(M + 1)]

        # Beta^M(b) = 1
        for j in range(self.L):
            betas[M][j] = 1

        for ridx in range(M-1, -1, -1):
            for cidx in range(self.L):
                lower_sum = 0
                if ridx != 0:
                    for lower_idx in range(self.L):
                        lower_sum += betas[ridx+1][lower_idx]*self.A[cidx][lower_idx]*self.O[lower_idx][x[ridx]]
                else:
                    for lower_idx in range(self.L):
                        lower_sum += betas[ridx+1][lower_idx]*self.A_start[cidx]*self.O[lower_idx][x[ridx]]
                betas[ridx][cidx] = lower_sum

        return betas

    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to D - 1. In other words, a list of
                        lists.
            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to L - 1. In other words, a list of
                        lists.
                        Note that the elements in X line up with those in Y.
        '''
        # Calculate each element of A using the M-step formulas.
        num_trng_points = len(Y)

        for ridx in range(self.L):
            for cidx in range(self.L):
                numer_count = 0
                denom_count = 0
                for i in range(num_trng_points):
                    for j in range(1, len(Y[i])):
                        if ridx == Y[i][j-1]:
                            denom_count += 1
                            if cidx == Y[i][j]:
                                numer_count += 1
                self.A[ridx][cidx] = numer_count / denom_count

        # Calculate each element of O using the M-step formulas.
        for ridx in range(self.L):
            for cidx in range(self.D):
                numer_count = 0
                denom_count = 0
                for i in range(num_trng_points):
                    for j in range(len(Y[i])):
                        if ridx == Y[i][j]:
                            denom_count += 1
                            if cidx == X[i][j]:
                                numer_count += 1
                self.O[ridx][cidx] = numer_count / denom_count


    def unsupervised_learning(self, X):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.
        '''

        num_iter = 10
        num_trng_points = len(X)


        for iter_counter in range(num_iter):
            # E-STEP
            # Create marginal probability matrix P(y=a, x) with size N x (M+1) x L
            M1 = [[[0. for k in range(self.L)] for i in range(len(X[j])+1)] for j in range(num_trng_points)]
            for ridx in range(num_trng_points):
                alphas = self.forward(X[ridx])
                betas = self.backward(X[ridx])
                for cidx in range(1, len(X[ridx]) + 1):
                    denom_sum = 0
                    for a in range(self.L):
                        denom_sum += alphas[cidx][a]*betas[cidx][a]
                    for a in range(self.L):
                        M1[ridx][cidx][a] = alphas[cidx][a]*betas[cidx][a] / denom_sum


            # Create marginal probability matrix P(yj=a, y(j+1)=b, x) with size N x M x L x L
            M2 = [[[[0. for l in range(self.L)] for k in range(self.L)] for i in range(len(X[j]) + 1)] for j in range(num_trng_points)]
            for ridx in range(num_trng_points):
                alphas = self.forward(X[ridx])
                betas = self.backward(X[ridx])
                for cidx in range(1, len(X[ridx])):
                    denom_sum = 0
                    for cur_idx in range(self.L):
                        for next_idx in range(self.L):
                            denom_sum += alphas[cidx][cur_idx]*self.O[next_idx][X[ridx][cidx]]* \
                                         self.A[cur_idx][next_idx]*betas[cidx+1][next_idx]

                    for cur_idx in range(self.L):
                        for next_idx in range(self.L):
                            M2[ridx][cidx][cur_idx][next_idx] = alphas[cidx][cur_idx]*self.O[next_idx][X[ridx][cidx]]* \
                                                         self.A[cur_idx][next_idx]*betas[cidx+1][next_idx] / denom_sum

            for ridx in range(num_trng_points):

            # M-STEP
            # Update transition matrix A
            for ridx in range(self.L):
                for cidx in range(self.L):
                    numer_sum = 0
                    denom_sum = 0
                    for i in range(num_trng_points):
                        for j in range(1, len(X[i])):
                            numer_sum += M2[i][j][ridx][cidx]
                            denom_sum += M1[i][j][ridx]
                    self.A[ridx][cidx] = numer_sum / denom_sum

            # Update obsevation matrix O
            for ridx in range(self.L):
                for cidx in range(self.D):
                    numer_sum = 0
                    denom_sum = 0
                    for i in range(num_trng_points):
                        for j in range(1, len(X[i])+1):
                            if X[i][j-1] == cidx:
                                numer_sum += M1[i][j][ridx]
                            denom_sum += M1[i][j][ridx]
                    self.O[ridx][cidx] = numer_sum / denom_sum



    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random.

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a string.
        '''

        emission = ''
        cur_state = random.randint(0, self.L - 1)
        for idx in range(M):
            new_char = np.random.choice(self.D, 1, p=self.O[cur_state])[0]
            cur_state = np.random.choice(self.L, 1, p=self.A[cur_state])[0]
            emission += str(new_char)


        return emission

    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''
        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the output sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any output sequence, i.e. the
        # probability of x.
        prob = sum(alphas[-1])
        return prob

    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''
        betas = self.backward(x)

        # beta_j(0) gives the probability of the output sequence. Summing
        # this over all states and then normalizing gives the total
        # probability of x paired with any output sequence, i.e. the
        # probability of x.
        prob = sum(betas[0]) / self.L

        return prob

def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learing.

    Arguments:
        X:          A list of variable length emission sequences
        Y:          A corresponding list of variable length state sequences
                    Note that the elements in X line up with those in Y
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)

    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrices A and O.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.
        n_states:   Number of hidden states to use in training.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrices A and O.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X)

    return HMM
