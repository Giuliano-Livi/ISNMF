import numpy as np
import matplotlib.pyplot as plt
import itertools
from itertools import permutations

class ISNMF:
    #function to initialize the model
    def __init__(self, V, r, beta, gamma, mu, epsilon, t_max):
        """
        n = number of muscles
        r = number of synergies
        k = number of samples
        beta    : Regularization parameter for basis matrix W
        gamma   : Regularization parameter for activation matrix H
        mu      : Forgetting factor (discounts old data)
        epsilon : Convergence threshold
        t_max   : Maximum number of iterations per update
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
        self.W = np.array((self.n,self.r))
        self.H = np.array((self.r,self.k))
        self.W = np.maximum(0, np.random.normal(loc=self.V_mean, scale=1, size=(self.n,self.r)))
        self.H = np.maximum(0, np.random.normal(loc=self.V_mean, scale=1, size=(self.r,self.k)))
        self.A = np.zeros((self.n, self.r))  #Accumulator for forgetting mechanism
        self.B = np.zeros((self.r, self.r))  #Accumulator for forgetting mechanism


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
            '''
            if np.any(self.gamma *(self.H) ** -0.5 != 0):
                print("no zero element at update {}".format(t))
                denominator_H = self.W.T @ self.W @ self.H + self.gamma*(self.H) ** -0.5
            else:
            '''
            denominator_H = self.W.T @ self.W @ self.H + self.gamma*(self.H) * 1e-2 #+ self.gamma*(self.H) ** -0.5
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


# function to find the minimum RMSE value over all the permutations of the rows of H matrix
def min_rmse_over_permutations(A, B):
    best_rmse = float('inf')
    for perm in permutations(range(A.shape[0])):
        B_perm = B[list(perm), :]
        rmse_val = np.sqrt(np.mean((A - B_perm)**2))
        if rmse_val < best_rmse:
            best_rmse = rmse_val
    return best_rmse

# testing of the algorithm using only 1 synergy
r = 1

# definition of the 4 shapes of W
W_top_triangular = np.array([1, 6/7, 5/7, 4/7, 3/7, 2/7, 1/7, 0]).reshape(8,1)    # triangular shape with max value in the first position
W_bottom_triangular = np.array([0, 1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 1]).reshape(8,1) # triangular shape with max value in the last position
W_center_triangular = np.array([0, 1/3, 2/3, 1, 1, 2/3, 1/3, 0]).reshape(8,1)     # triangular shape with max value in the middle
W_low_center_triangular = np.array([1, 2/3, 1/3, 0, 0, 1/3, 2/3, 1]).reshape(8,1) # triangular shape with max value at the top and bottom and lower value in the middle 

W_base_list = [W_top_triangular, W_bottom_triangular, W_center_triangular, W_low_center_triangular] #list of all the basic components of W matrix

# definition of the H used in case of 1 synergy
H_1 = np.concatenate([np.full(200, 1), np.linspace(1, 0, 200), np.full(200, 0), np.linspace(0, 1, 200), np.full(200, 1)]).reshape(1, 1000)

#finding the RMSE value for all the cases
rmse_list = []
for element in W_base_list:
    rmse_value = 0
    for i in range(1000):
        V_test = element @ H_1
        model = ISNMF(V_test, r, beta=5, gamma=5, mu=0.95, epsilon=1e-5, t_max=200)
        W_found, H_found = model.update(V_test)
        rmse_value += np.sqrt(np.mean((H_1 - H_found) ** 2))
    rmse_list.append(rmse_value/1000)
    print(rmse_value/1000)

# plotting of the results for the case of 1 synergy
# Labels for each case
labels = ['Top', 'Bottom', 'Center', 'Low Center']
# Create bar plot
plt.bar(labels, rmse_list, color='skyblue', edgecolor='black')
plt.ylabel("RMSE (normalized)")
plt.title("RMSE Comparison for Triangular Activations")
plt.ylim(0, 0.1)
plt.show()
print("done for 1 synergy")


# testing of the algorithm using 2 synergies
r = 2

# composing all the combinations of the W matrix
W_cases = {}  # dictionary to store combinations
for i, j in itertools.product(range(4), repeat=2):
    key = f"W_{i+1}_{j+1}"
    W_cases[key] = np.hstack((W_base_list[i], W_base_list[j]))

# definition of the H used in case of 2 synergies
H_2_2 = np.concatenate([np.full(200, 0), np.linspace(0, 1, 200), np.full(200, 1), np.linspace(1, 0, 200), np.full(200, 0)]).reshape(1, 1000)
H_2 = np.vstack((H_1, H_2_2)) # 1,2 synergy

#finding the RMSE value for all the cases
rmse_list = []
for key, value in W_cases.items():
    rmse_value = 0
    for i in range(1000):
        V_test = value @ H_2
        model = ISNMF(V_test, r, beta=5, gamma=5, mu=0.95, epsilon=1e-5, t_max=200)
        W_found, H_found = model.update(V_test)
        H_found_swapped = H_found[[1, 0], :] # Swap the rows of H_found to match the order of H_2
        rmse_value += min(np.sqrt(np.mean((H_2 - H_found) ** 2)), np.sqrt(np.mean((H_2 - H_found_swapped) ** 2)))
    rmse_list.append(rmse_value/1000)
    print(rmse_value/1000)

# Create bar plot
plt.bar(W_cases.keys(), rmse_list, color='skyblue', edgecolor='black')
plt.ylabel("RMSE (normalized)")
plt.title("RMSE Comparison for Triangular Activations")
plt.ylim(0, 0.7)
plt.show()

print("done for 2 synergies")



# testing the algorithm using 3 synergies
r = 3

# composing all the combinations of the W matrix
W_cases = {}  # dictionary to store combinations

# Generate all possible combinations (4 x 4 x 4 = 64 total)
for i, j, k in itertools.product(range(4), repeat=3):
    key = f"W_{i+1}_{j+1}_{k+1}"
    W_cases[key] = np.hstack((W_base_list[i], W_base_list[j], W_base_list[k]))

# definition of the H matrix used in case of 3 synergies
H_3_1 = np.concatenate([np.full(76,1), np.linspace(1,0,76), np.full(684,0), np.linspace(0,1,76), np.full(76,1)])
H_3_2 = np.concatenate([np.full(152,0), np.linspace(0,1,76), np.full(76,1), np.linspace(1,0,76), np.full(228,0), np.linspace(0,1,76), np.full(76,1),
                         np.linspace(1,0,76), np.full(152,0)])
H_3_3 = np.concatenate([np.full(380,0), np.linspace(0,1,76), np.full(76,1), np.linspace(1,0,76), np.full(380,0)])
H_3 = np.vstack((H_3_1, H_3_2, H_3_3))

rmse_list = []
for key, value in W_cases.items():
    rmse_value = 0
    for i in range(1000):
        V_test = value @ H_3
        model = ISNMF(V_test, r, beta=5, gamma=5, mu=0.95, epsilon=1e-5, t_max=200)
        W_found, H_found = model.update(V_test)
        rmse_value += min_rmse_over_permutations(H_3, H_found)
    rmse_list.append(rmse_value/1000)
    print(rmse_value/1000)

# Create bar plot
plt.bar(W_cases.keys(), rmse_list, color='skyblue', edgecolor='black')
plt.xticks(rotation=45)
plt.tight_layout()
plt.ylabel("RMSE (normalized)")
plt.title("RMSE Comparison for Triangular Activations")
plt.ylim(0, 0.7)
plt.show()

print("done for 3 synergies")