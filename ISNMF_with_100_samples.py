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
            denominator_H = self.W.T @ self.W @ self.H + self.gamma*(self.H) * 1e-1 #+ self.gamma*(self.H) ** -0.5
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




#definition of the number of synergies
r = 2

#initialization of the model
W_test = np.array([[1,   0],
                   [0,   1],
                   [0.5, 0],
                   [0, 0.5]])

x = np.linspace(0, 2 * np.pi, 100)
wave1 = np.maximum(np.sin(x), 0)
wave2 = np.maximum(np.sin(x + np.pi), 0)
H_test = np.array([wave1, wave2])
V_test = W_test @ H_test
model = ISNMF(V_test, r, beta=5, gamma=5, mu=0.95, epsilon=1e-5, t_max=200)

#graphical representation section
for i in range(2):
    W_found, H_found = model.update(model.V)
    plt.figure(figsize=(8, 9)) 
    plt.subplot(2,1,1)

    #graphical representation of the W_found compared with the original W matrix
    plt.subplot(2,1,1)
    for j in range(W_found.shape[0]):
        x = np.linspace(0, W_found.shape[1] , W_found.shape[1])
        plt.plot(x, W_test[j], 'o-', color='blue')
        plt.plot(x, W_found[j], 'o-', color='red')
    plt.title("components of the W matrix(activation matrix)")
    plt.xlabel("sinergy")
    plt.ylabel("muscles activation")
    plt.ylim(-0.1, 2)

    #graphical representation of the H_found compared with the original H matrix
    plt.subplot(2,1,2)
    for j in range(H_found.shape[0]):
        x = np.linspace(0, model.V.shape[1] , model.V.shape[1])
        plt.plot(x, H_test[j], 'o-', color='blue')
        plt.plot(x, H_found[j], 'o-', color='red')
    plt.title("components of the H matrix(activation matrix)")
    plt.xlabel("samples")
    plt.ylabel("sinergy activations")
    plt.tight_layout()
    plt.show()



#section in which I add the 3rd synergy to the model
W_test = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 0]])
x = np.linspace(0, 2 * np.pi, 100)
wave1 = np.maximum(np.sin(x), 0)
wave2 = np.maximum(np.sin(x + np.pi/2), 0)
wave3 = np.maximum(np.sin(x + np.pi), 0)
H_test = np.array([wave1, wave2, wave3])
V_test = W_test @ H_test

#graphical representation section
for i in range(2):
    if i == 0:
        W_found, H_found = model.update(V_test, string="add")
    else:
        W_found, H_found = model.update(V_test)
    plt.figure(figsize=(8, 9)) 
    plt.subplot(2,1,1)

    #graphical representation of the W_found compared with the original W matrix
    plt.subplot(2,1,1)
    for j in range(W_found.shape[0]):
        x = np.linspace(0, W_found.shape[1] , W_found.shape[1])
        plt.plot(x, W_test[j], 'o-', color='blue')
        plt.plot(x, W_found[j], 'o-', color='red')
    plt.title("components of the W matrix(activation matrix)")
    plt.xlabel("sinergy")
    plt.ylabel("muscles activation")
    plt.ylim(-0.1, 2)

    #graphical representation of the H_found compared with the original H matrix
    plt.subplot(2,1,2)
    for j in range(H_found.shape[0]):
        x = np.linspace(0, model.V.shape[1] , model.V.shape[1])
        plt.plot(x, H_test[j], 'o-', color='blue')
        plt.plot(x, H_found[j], 'o-', color='red')
    plt.title("components of the H matrix(activation matrix)")
    plt.xlabel("samples")
    plt.ylabel("sinergy activations")
    plt.tight_layout()
    plt.show()



#section in which I add the 4th synergy to the model
W_test = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
x = np.linspace(0, 2 * np.pi, 100)
wave1 = np.maximum(np.sin(x), 0)
wave2 = np.maximum(np.sin(x + np.pi/2), 0)
wave3 = np.maximum(np.sin(x + np.pi), 0)
wave4 = np.maximum(np.sin(x + 3*np.pi/2), 0)
H_test = np.array([wave1, wave2, wave3])
V_test = W_test @ H_test

#graphical representation section
for i in range(2):
    if i == 0:
        W_found, H_found = model.update(V_test, string="add")
    else:
        W_found, H_found = model.update(V_test)
    plt.figure(figsize=(8, 9)) 

    #graphical representation of the W_found compared with the original W matrix
    plt.subplot(2,1,1)
    for j in range(W_found.shape[0]):
        x = np.linspace(0, W_found.shape[1] , W_found.shape[1])
        plt.plot(x, W_test[j], 'o-', color='blue')
        plt.plot(x, W_found[j], 'o-', color='red')
    plt.title("components of the W matrix(activation matrix)")
    plt.xlabel("sinergy")
    plt.ylabel("muscles activation")
    plt.ylim(-0.1, 2)

    #graphical representation of the H_found compared with the original H matrix
    plt.subplot(2,1,2)
    for j in range(H_found.shape[0]):
        x = np.linspace(0, model.V.shape[1] , model.V.shape[1])
        plt.plot(x, H_test[j], 'o-', color='blue')
        plt.plot(x, H_found[j], 'o-', color='red')
    plt.title("components of the H matrix(activation matrix)")
    plt.xlabel("samples")
    plt.ylabel("sinergy activations")
    plt.tight_layout()
    plt.show()