import numpy as np
import matplotlib.pyplot as plt
import scipy.io

#function to read the emg signals present in a .mat file and return them as a list
def load_ninapro_data(mat_file):
    emg = scipy.io.loadmat(mat_file)['emg']
    trials = []
    trials.append({'emg': emg})
    return trials


def generate_W(trial, num_synergies, n_samples):
    selected_segments = []
    trial_emg = trial[0]['emg'][:, 0]  # select first channel
    for _ in range(num_synergies):
        start = np.random.randint(0, len(trial_emg) - n_samples)
        segment = trial_emg[start:start+n_samples]
        selected_segments.append(segment)
    # Stack the segments to form the W matrix (num_synergies x n_samples)
    W = np.vstack(selected_segments)
    return W

#noise generation function
def generate_noise(noise_std, m, n):
    E = np.random.normal(0, noise_std, (m, n)) # (center of distribution, standard deviation, size of the matrix)
    g_E = np.maximum(0, E)
    return g_E


class ISNMF:
    #function to initialize the model
    def __init__(self, V, r, beta, gamma, mu, epsilon, t_max):
        """
        n = number of muscles
        r = number of synergies
        k = number of samples
        beta  : Regularization parameter for basis matrix W
        gamma : Sparsity constraint for encoding matrix H
        mu    : Forgetting factor (discounts old data)
        epsilon : Convergence threshold
        t_max  : Maximum number of iterations per update
        """
        self.V = V
        self.n = V.shape[0]
        self.k = V.shape[1]
        self.r = r
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
        self.epsilon = epsilon
        self.t_max = t_max

        #initialize the basis matrix ufg values from max(0,N(V,1))
        self.V_mean = np.mean(self.V)
        self.W = np.array((self.n,r))
        self.H = np.array((r,self.k))
        self.W = np.maximum(0, np.random.normal(loc=self.V_mean, scale=1, size=(self.n,r)))
        self.H = np.maximum(0, np.random.normal(loc=self.V_mean, scale=1, size=(r,self.k)))
        self.A = np.zeros((self.n, r))  #Accumulator for forgetting mechanism
        self.B = np.zeros((r, r))  #Accumulator for forgetting mechanism


        #initial reconstuction error
        self.e_0 = np.linalg.norm(self.V - (self.W @ self.H), 'fro') ** 2
        #number of the update
        self.m = 0

    #function to update the model when new samples arrives
    def update(self, V, string=""):
        self.V = V
        self.m += 1
        
        #update of the historical avarage value
        if self.m > 1:
            V_mean_previous = self.V_mean
            self.V_mean = np.mean(self.V)
            self.V_mean = (V_mean_previous + self.V_mean) / 2
            self.H = np.maximum(0, np.random.normal(loc=self.V_mean, scale=1, size=(self.r,self.k)))
            self.e_0 = np.linalg.norm(self.V - (self.W @ self.H), 'fro') ** 2
        
        #addiction of 1 components (r + 1)
        if string == "add":
            self.r += 1
            self.A = np.hstack((self.A, np.zeros((self.n, 1))))
            self.B = np.pad(self.B, ((0, 1), (0, 1)), mode='constant', constant_values=0)
            self.W = np.hstack((self.W, np.random.normal(loc=self.V_mean, scale=1, size=(self.n, 1))))
            self.H = np.vstack((self.H, np.random.normal(loc=self.V_mean, scale=1, size=(1, self.k))))
            self.e_0 = np.linalg.norm(self.V - (self.W @ self.H), 'fro') ** 2
        t = 0
        e_prev = 0

        while True:
            t += 1
            
            # Update W
            numerator_W = self.mu * self.A + self.V @ self.H.T
            denominator_W = self.mu * self.W @ self.B + self.W @ self.H @ self.H.T + (self.mu * (1 - (self.mu ** self.m)) / (1 - self.mu)) * self.beta * self.W
            self.W *= numerator_W / (denominator_W + 1e-10)
            self.W = np.maximum(self.W, self.epsilon)
            
            # Update H
            numerator_H = self.W.T @ self.V
            denominator_H = self.W.T @ self.W @ self.H + self.gamma*(self.H) *1e-1
            self.H *= numerator_H / (denominator_H + 1e-10)
            self.H = np.maximum(self.H, self.epsilon)
            
            # Compute error
            e_t = np.linalg.norm(self.V - self.W @ self.H, 'fro') ** 2

            # Convergence check
            if abs(e_t - e_prev) / self.e_0 < self.epsilon or t > self.t_max:
                break
            e_prev = e_t

        #update of A and B matrices
        self.A = self.mu * self.A + self.V @ self.H.T
        self.B = self.mu * self.B + self.H @ self.H.T
        return self.W, self.H


#here I'll test the algorithm using synthetic data
m = 12   #number of channels/muscles
n = 1000 #number of samples
r = 4    #number of synergies

M = np.array((m,n)) #matrix of the synthetic data
S = np.array((m,r)) #matrix of the synthetic synergies
W = np.array((r,n)) #weighting function matrix

#generate the S matrix of synthetic data
S = np.random.rand(m,r)

#exctract existing synergies from the Ninapro dataset
trials = load_ninapro_data('synthetic/S1_A1_E1.mat') 
W = generate_W(trials, r, n)

#graphical representation of the synergies extracted from the Ninapro dataset
x = np.linspace(0, 1000, 1000)
for i in range(4):
    plt.plot(x, W[i], linestyle='-', color='blue')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()


#create the matrix of the synthetic data
M = S @ W + generate_noise(0.01, m, n)

#using the ISNMF algorithm to extract the synergies
model = ISNMF(M, r, beta=5, gamma=5, mu=0.4, epsilon=1e-5, t_max=200)
x = np.linspace(0, 1000, 1000)


for i in range(4):
    W_found, H_found = model.update(model.V)
    #graphical representation of all the founded synergies and original once overlapped
    plt.plot(x, W[0], linestyle='-', color='blue')
    plt.plot(x, H_found[0], linestyle='-', color='red')
    plt.plot(x, W[1], linestyle='-', color='blue')
    plt.plot(x, H_found[1], linestyle='-', color='red')
    plt.plot(x, W[2], linestyle='-', color='blue')
    plt.plot(x, H_found[2], linestyle='-', color='red')
    plt.plot(x, W[3], linestyle='-', color='blue')
    plt.plot(x, H_found[3], linestyle='-', color='red')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    print("the starting S is =")
    print(S)
    print("the found S is =")
    print(W_found)
    plt.show()

    #graphical representation of the 1st synergy at the 1st update
    plt.plot(x, W[0], linestyle='-', color='blue')
    plt.plot(x, H_found[0], linestyle='-', color='red')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()

    #graphical representation of the 2nd synergy at the 1st update
    plt.plot(x, W[1], linestyle='-', color='blue')
    plt.plot(x, H_found[1], linestyle='-', color='red')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()

    #graphical representation of the 3rd synergy at the 1st update
    plt.plot(x, W[2], linestyle='-', color='blue')
    plt.plot(x, H_found[2], linestyle='-', color='red')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()

    #graphical representation of the 4th synergy at the 1st update
    plt.plot(x, W[3], linestyle='-', color='blue')
    plt.plot(x, H_found[3], linestyle='-', color='red')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()
