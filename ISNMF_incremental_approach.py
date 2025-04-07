import rosbag
import numpy as np
import matplotlib.pyplot as plt

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
        self.previous_V = np.zeros((self.n, self.k))  # Initialize previous_V to zeros

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
    def training(self, V, string=""):
        self.V = V
        self.m += 1

        #change dimension of the V matrix in case it is different from the first one
        if self.V.shape[1] != self.k:
            self.k = self.V.shape[1]
            self.H = np.maximum(0, np.random.normal(loc=self.V_mean, scale=1, size=(self.r,self.k)))

        
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
            denominator_H = self.W.T @ self.W @ self.H + self.gamma*(self.H) * 1e-3
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
        self.previous_V = self.V
        return self.W, self.H
        
    
    def test(self, V):
        self.V = V
        #change dimension of the V matrix in case it is different from the first one
        if self.V.shape[1] != self.k:
            self.k = self.V.shape[1]
            self.H = np.maximum(0, np.random.normal(loc=self.V_mean, scale=1, size=(self.r,self.k)))

        self.H = np.linalg.pinv(self.W) @ self.V

        order = []
        #check if a shift has happened in the brecelet
        for i in range(self.V.shape[0]):
            acc = np.zeros((self.V.shape[0]))
            for j in range(self.V.shape[0]):
                for k in range(self.V.shape[1]):
                    if self.previous_V.shape[1] <= k:
                        break
                    acc[j] += np.abs(self.V[i][k] - self.previous_V[j][k])
            order.append(np.argmin(acc))
        if order == [1, 2, 3, 4, 5, 6, 7, 0] or order == [7, 0, 1, 2, 3, 4, 5, 6]:
            print("shift detected, model recalibration")
            model.training(self.V)
            return self.H
                    
        self.previous_V = self.V
        return self.H


#function to extract the matrix M from the dataset
def extract_M_matrix_from_dataset(bag_path):
    # Initialize lists to store timestamps and EMG data
    timestamps = []
    emg_data = []

    # Open the ROS bag file and read messages from the 'emg_rms' topic
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, rawdata, timestamp in bag.read_messages(topics=['emg_rms']):
            emg_data.append(rawdata.data)  # Extract EMG RMS values from the message
    # Convert lists to NumPy arrays for easier processing
    emg_data = np.array(emg_data)
    return emg_data.T

#function to filter the data
def avaraging(array, point_to_avarage):
    i_prev = 0
    i = 0
    while i < array.shape[0]:
        i = i + point_to_avarage
        mean = np.mean(array[i_prev : i])
        array[i_prev : i] = mean
        i_prev = i
    return array

#noise generation function
def generate_noise(noise_std, m, n):
    E = np.random.normal(0, noise_std, (m, n)) # (center of distribution, standard deviation, size of the matrix)
    g_E = np.maximum(0, E)
    return g_E

#model training using the rep0_power.bag
M = extract_M_matrix_from_dataset('dataset/rep0_ulnar.bag')
r = 3
model = ISNMF(M, r, beta=32, gamma=32, mu=0.95, epsilon=1e-5, t_max=200)

#training the model on rep0_power.bag
for i in range(2):
    W_found, H_found = model.training(M)
    #graphical representation of the M input matrix
    plt.figure(figsize=(8, 9)) 
    plt.subplot(3,1,1)
    for j in range(model.V.shape[0]):
        x = np.linspace(0, M.shape[1] , M.shape[1])
        plt.plot(x, model.V[j], linestyle='-', label='muscle {}'.format(j))
    plt.title("components of the M input matrix")
    plt.xlabel("samples")
    plt.ylabel("muscles activations")
    plt.legend(loc='best', fontsize='small', markerscale=1)
    #graphical representation of the W_found matrix
    plt.subplot(3,1,2)
    for j in range(W_found.shape[0]):
        x = np.linspace(0, W_found.shape[1] , W_found.shape[1])
        plt.plot(x, W_found[j], 'o-')
    plt.title("components of the W matrix(activation matrix)")
    plt.xlabel("sinergy")
    plt.ylabel("muscles activation")
    plt.ylim(-0.1, 7)
    #graphical representation of the H_found matrix
    plt.subplot(3,1,3)
    for j in range(H_found.shape[0]):
        x = np.linspace(0, M.shape[1] , M.shape[1])
        H_found[j] = avaraging(H_found[j], 30)
        plt.plot(x, H_found[j], linestyle='-')
    plt.title("components of the H matrix(activation matrix)")
    plt.xlabel("samples")
    plt.ylabel("sinergy activations")
    plt.tight_layout()
    plt.show()

#test of the model using the rep1_power.bag
M_test = extract_M_matrix_from_dataset('dataset/rep1_ulnar.bag')
H_found = model.test(M_test)
#graphical representation of the M_test input matrix
plt.figure(figsize=(8, 9)) 
plt.subplot(2,2,1)
for j in range(model.V.shape[0]):
    x = np.linspace(0, M_test.shape[1] , M_test.shape[1])
    plt.plot(x, M_test[j], linestyle='-', label='muscle {}'.format(j))
plt.title("components of the M input matrix")
plt.xlabel("samples")
plt.ylabel("muscles activations")
plt.legend(loc='best', fontsize='small', markerscale=1)
#graphical representation of the W_found matrix
plt.subplot(2,2,2)
for j in range(H_found.shape[0]):
    x = np.linspace(0, M_test.shape[1] , M_test.shape[1])
    H_found[j] = avaraging(H_found[j], 30)
    plt.plot(x, H_found[j], linestyle='-')
plt.title("components of the H matrix(activation matrix)")
plt.xlabel("samples")
plt.ylabel("sinergy activations")



#test of the model using the rep1_power.bag with shifted matrix
shifted_M = np.roll(M_test, shift=-1, axis=0)
H_found_shifted = model.test(shifted_M)
#graphical representation of the M_test input matrix
plt.subplot(2,2,3)
for j in range(model.V.shape[0]):
    x = np.linspace(0, M_test.shape[1] , M_test.shape[1])
    plt.plot(x, shifted_M[j], linestyle='-', label='muscle {}'.format(j))
plt.title("components of the M input matrix")
plt.xlabel("samples")
plt.ylabel("muscles activations")
plt.legend(loc='best', fontsize='small', markerscale=1)
#graphical representation of the W_found matrix
plt.subplot(2,2,4)
for j in range(H_found_shifted.shape[0]):
    x = np.linspace(0, M_test.shape[1] , M_test.shape[1])
    H_found_shifted[j] = avaraging(H_found_shifted[j], 30)
    plt.plot(x, H_found_shifted[j], linestyle='-')
plt.title("components of the H matrix(activation matrix)")
plt.xlabel("samples")
plt.ylabel("sinergy activations")
plt.tight_layout()
plt.show()