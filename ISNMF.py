import numpy as np
import matplotlib.pyplot as plt

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
        self.W = np.array((n,r))
        self.H = np.array((r,k))
        self.W = np.maximum(0, np.random.normal(loc=self.V_mean, scale=1, size=(n,r)))
        self.H = np.maximum(0, np.random.normal(loc=self.V_mean, scale=1, size=(r,k)))
        self.A = np.zeros((n, r))  #Accumulator for forgetting mechanism
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




#definition of the dimensions of the matrices
r = 2; n = 4; k = 10

#initialization of the model
W_test = np.array([[1, 0],
                   [1, 0],
                   [0, 1],
                   [0, 1]])
H_test = np.array([[0.00000000e+00, 6.42787610e-01, 9.84807753e-01, 8.66025404e-01, 3.42020143e-01, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 3.42020143e-01, 8.66025404e-01, 9.84807753e-01, 6.42787610e-01, 0.00000000e+00]])


V_test = W_test @ H_test
model = ISNMF(V_test, r, beta=5, gamma=5, mu=0.4, epsilon=1e-5, t_max=200)
x = np.linspace(0, 2 * np.pi, 10)

#first update verification
W_found, H_found = model.update(model.V)
#graphical representation
x = np.linspace(0, 2 * np.pi, 10)
plt.plot(x, H_test[0], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test[1], marker='o', linestyle='-', color='blue')
plt.plot(x, H_found[0], marker='o', linestyle='-', color='red')
plt.plot(x, H_found[1], marker='o', linestyle='-', color='red')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("1st update")
print("W of the 1st update is = ")
print(W_found)
print("H of the 1st update is = ")
print(H_found)
plt.show()




#second update verification
W_test_2 = np.array([[1, 0],
                     [1, 0],
                     [0, 1],
                     [0, 1]])
H_test_2 = np.array([[0.00000000e+00, 6.42787610, 9.84807753, 8.66025404, 3.42020143, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 3.42020143, 8.66025404, 9.84807753, 6.42787610, 0.00000000e+00]])
V_test_2 = W_test_2 @ H_test_2
W_found_2, H_found_2 = model.update(V_test_2)
#graphical representation
plt.plot(x, H_test_2[0], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test_2[1], marker='o', linestyle='-', color='blue')
plt.plot(x, H_found_2[0], marker='o', linestyle='-', color='red')
plt.plot(x, H_found_2[1], marker='o', linestyle='-', color='red')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("2nd update")
print("W of the 2nd update is = ")
print(W_found_2)
print("H of the 2nd update is = ")
print(H_found_2)
plt.show()





#third update verification
W_test_3 = np.array([[1, 0],
                     [1, 0],
                     [0, 1],
                     [0, 1]])
H_test_3 = np.array([[0.00000000e+00, 6.427876103e+01, 9.84807753e+01, 8.66025404e+01, 3.42020143e+01, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 3.42020143e+01, 8.66025404e+01, 9.84807753e+01, 6.42787610e+01, 0.00000000e+00]])
V_test_3 = W_test_3 @ H_test_3
W_found_3, H_found_3 = model.update(V_test_3)
#graphical representation
plt.plot(x, H_test_3[0], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test_3[1], marker='o', linestyle='-', color='blue')
plt.plot(x, H_found_3[0], marker='o', linestyle='-', color='red')
plt.plot(x, H_found_3[1], marker='o', linestyle='-', color='red')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("3rd update")
print("W of the 3rd update is = ")
print(W_found_3)
print("H of the 3rd update is = ")
print(H_found_3)
plt.show()




#fourth update verification
W_test_4 = np.array([[1, 0],
                     [1, 0],
                     [0, 1],
                     [0, 1]])
H_test_4 = np.array([[0.00000000e+00, 6.427876103e-01, 9.84807753e-01, 8.66025404e-01, 3.42020143e-01, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 3.42020143e-01, 8.66025404e-01, 9.84807753e-01, 6.42787610e-01, 0.00000000e+00]])
V_test_4 = W_test_4 @ H_test_4
W_found_4, H_found_4 = model.update(V_test_4)
#graphical representation
plt.plot(x, H_test_4[0], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test_4[1], marker='o', linestyle='-', color='blue')
plt.plot(x, H_found_4[0], marker='o', linestyle='-', color='red')
plt.plot(x, H_found_4[1], marker='o', linestyle='-', color='red')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("4th update")
print("W of the 4th update is = ")
print(W_found_4)
print("H of the 4th update is = ")
print(H_found_4)
plt.show()




#fifth update verification
W_test_5 = np.array([[1, 0],
                     [1, 0],
                     [0, 1],
                     [0, 1]])
H_test_5 = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                     [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
V_test_5 = W_test_5 @ H_test_5
W_found_5, H_found_5 = model.update(V_test_5)
#graphical representation
plt.plot(x, H_test_5[0], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test_5[1], marker='o', linestyle='-', color='blue')
plt.plot(x, H_found_5[0], marker='o', linestyle='-', color='red')
plt.plot(x, H_found_5[1], marker='o', linestyle='-', color='red')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("5th update")
print("W of the 5th update is = ")
print(W_found_5)
print("H of the 5th update is = ")
print(H_found_5)
plt.show()



#sixth update verification
W_test_6 = np.array([[1, 0, 0],
                     [1, 0, 0],
                     [0, 1, 0],
                     [0, 1, 1]])
H_test_6 = np.array([[0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1.5, 1.5, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1.7, 1.7, 0]])
V_test_6 = W_test_6 @ H_test_6
W_found_6, H_found_6 = model.update(V_test_6, "add")
#graphical representation
plt.plot(x, H_test_6[0], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test_6[1], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test_6[2], marker='o', linestyle='-', color='blue')
plt.plot(x, H_found_6[0], marker='o', linestyle='-', color='red')
plt.plot(x, H_found_6[1], marker='o', linestyle='-', color='red')
plt.plot(x, H_found_6[2], marker='o', linestyle='-', color='red')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("6th update")
print("W of the 6th update is = ")
print(W_found_6)
print("H of the 6th update is = ")
print(H_found_6)
# Show the graph
plt.show()



#seventh update verification
W_test_7 = np.array([[1, 0, 0],
                     [1, 0, 0],
                     [0, 1, 0],
                     [0, 1, 1]])
H_test_7 = np.array([[0, 1e+1, 1e+1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1e+1, 1e+1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1e+1, 1e+1, 0]])
V_test_7 = W_test_7 @ H_test_7
W_found_7, H_found_7 = model.update(V_test_7)
#graphical representation
plt.plot(x, H_test_7[0], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test_7[1], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test_7[2], marker='o', linestyle='-', color='blue')
plt.plot(x, H_found_7[0], marker='o', linestyle='-', color='red')
plt.plot(x, H_found_7[1], marker='o', linestyle='-', color='red')
plt.plot(x, H_found_7[2], marker='o', linestyle='-', color='red')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("7th update")
print("W of the 7th update is = ")
print(W_found_7)
print("H of the 7th update is = ")
print(H_found_7)
# Show the graph
plt.show()



#eighth update verification
W_test_8 = np.array([[1, 0, 0],
                     [1, 0, 0],
                     [0, 1, 0],
                     [0, 1, 1]])
H_test_8 = np.array([[0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]])
V_test_8 = W_test_8 @ H_test_8
W_found_8, H_found_8 = model.update(V_test_8)
#graphical representation
plt.plot(x, H_test_8[0], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test_8[1], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test_8[2], marker='o', linestyle='-', color='blue')
plt.plot(x, H_found_8[0], marker='o', linestyle='-', color='red')
plt.plot(x, H_found_8[1], marker='o', linestyle='-', color='red')
plt.plot(x, H_found_8[2], marker='o', linestyle='-', color='red')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("8th update")
print("W of the 8th update is = ")
print(W_found_8)
print("H of the 8th update is = ")
print(H_found_8)
# Show the graph
plt.show()



#eighth update verification
W_test_9 = np.array([[1, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 1, 0]])
H_test_9 = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
V_test_9 = W_test_9 @ H_test_9
W_found_9, H_found_9 = model.update(V_test_9, "add")
#graphical representation
plt.plot(x, H_test_9[0], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test_9[1], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test_9[2], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test_9[3], marker='o', linestyle='-', color='blue')
plt.plot(x, H_found_9[0], marker='o', linestyle='-', color='red')
plt.plot(x, H_found_9[1], marker='o', linestyle='-', color='red')
plt.plot(x, H_found_9[2], marker='o', linestyle='-', color='red')
plt.plot(x, H_found_9[3], marker='o', linestyle='-', color='red')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("9th update")
print("W of the 9th update is = ")
print(W_found_9)
print("H of the 9th update is = ")
print(H_found_9)
# Show the graph
plt.show()



#tenth update verification
W_test_10 = np.array([[1, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 1, 0]])
H_test_10 = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
V_test_10 = W_test_10 @ H_test_10
W_found_10, H_found_10 = model.update(V_test_10)
#graphical representation
plt.plot(x, H_test_10[0], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test_10[1], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test_10[2], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test_10[3], marker='o', linestyle='-', color='blue')
plt.plot(x, H_found_10[0], marker='o', linestyle='-', color='red')
plt.plot(x, H_found_10[1], marker='o', linestyle='-', color='red')
plt.plot(x, H_found_10[2], marker='o', linestyle='-', color='red')
plt.plot(x, H_found_10[3], marker='o', linestyle='-', color='red')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("10th update")
print("W of the 10th update is = ")
print(W_found_10)
print("H of the 10th update is = ")
print(H_found_10)
# Show the graph
plt.show()



#eleventh update verification
W_test_11 = np.array([[1, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 1, 0]])
H_test_11 = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
V_test_11 = W_test_11 @ H_test_11
W_found_11, H_found_11 = model.update(V_test_11)
#graphical representation
plt.plot(x, H_test_11[0], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test_11[1], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test_11[2], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test_11[3], marker='o', linestyle='-', color='blue')
plt.plot(x, H_found_11[0], marker='o', linestyle='-', color='red')
plt.plot(x, H_found_11[1], marker='o', linestyle='-', color='red')
plt.plot(x, H_found_11[2], marker='o', linestyle='-', color='red')
plt.plot(x, H_found_11[3], marker='o', linestyle='-', color='red')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("11th update")
print("W of the 11th update is = ")
print(W_found_11)
print("H of the 11th update is = ")
print(H_found_11)
# Show the graph
plt.show()



#twelveth update verification
W_test_12 = np.array([[1, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 1, 0]])
H_test_12 = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
V_test_12 = W_test_12 @ H_test_12
W_found_12, H_found_12 = model.update(V_test_12)
#graphical representation
plt.plot(x, H_test_12[0], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test_12[1], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test_12[2], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test_12[3], marker='o', linestyle='-', color='blue')
plt.plot(x, H_found_12[0], marker='o', linestyle='-', color='red')
plt.plot(x, H_found_12[1], marker='o', linestyle='-', color='red')
plt.plot(x, H_found_12[2], marker='o', linestyle='-', color='red')
plt.plot(x, H_found_12[3], marker='o', linestyle='-', color='red')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("12th update")
print("W of the 12th update is = ")
print(W_found_12)
print("H of the 12th update is = ")
print(H_found_12)
# Show the graph
plt.show()



#thirteenth update verification
W_test_13 = np.array([[1, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 1, 0]])
H_test_13 = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
V_test_13 = W_test_13 @ H_test_13
W_found_13, H_found_13 = model.update(V_test_13)
#graphical representation
plt.plot(x, H_test_13[0], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test_13[1], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test_13[2], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test_13[3], marker='o', linestyle='-', color='blue')
plt.plot(x, H_found_13[0], marker='o', linestyle='-', color='red')
plt.plot(x, H_found_13[1], marker='o', linestyle='-', color='red')
plt.plot(x, H_found_13[2], marker='o', linestyle='-', color='red')
plt.plot(x, H_found_13[3], marker='o', linestyle='-', color='red')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("13th update")
print("W of the 13th update is = ")
print(W_found_13)
print("H of the 13th update is = ")
print(H_found_13)
# Show the graph
plt.show()


#fourteenth update verification
W_test_14 = np.array([[1, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 1, 0]])
H_test_14 = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
V_test_14 = W_test_14 @ H_test_14
W_found_14, H_found_14 = model.update(V_test_14)
#graphical representation
plt.plot(x, H_test_14[0], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test_14[1], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test_14[2], marker='o', linestyle='-', color='blue')
plt.plot(x, H_test_14[3], marker='o', linestyle='-', color='blue')
plt.plot(x, H_found_14[0], marker='o', linestyle='-', color='red')
plt.plot(x, H_found_14[1], marker='o', linestyle='-', color='red')
plt.plot(x, H_found_14[2], marker='o', linestyle='-', color='red')
plt.plot(x, H_found_14[3], marker='o', linestyle='-', color='red')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("14th update")
print("W of the 14th update is = ")
print(W_found_14)
print("H of the 14th update is = ")
print(H_found_14)
# Show the graph
plt.show()