import numpy as np
import matplotlib.pyplot as plt

# definition of the 4 shapes of W
W_top_triangular = np.array([1, 6/7, 5/7, 4/7, 3/7, 2/7, 1/7, 0])
W_bottom_triangular = np.array([0, 1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 1])
W_center_triangular = np.array([0, 1/3, 2/3, 1, 1, 2/3, 1/3, 0])
W_low_center_triangular = np.array([1, 2/3, 1/3, 0, 0, 1/3, 2/3, 1])

y_positions = np.arange(1, len(W_top_triangular) + 1)
plt.barh(y_positions, W_top_triangular)
plt.xlabel('Contribution to synergy')
plt.ylabel('EMG channel')
plt.yticks(y_positions)  # shows 1 through 8 on the y-axis
plt.title('Horizontal Bar Plot of W_top_triangular')
plt.gca().invert_yaxis()
plt.show()

plt.barh(y_positions, W_bottom_triangular)
plt.xlabel('Contribution to synergy')
plt.ylabel('EMG channel')
plt.yticks(y_positions)  # shows 1 through 8 on the y-axis
plt.title('Horizontal Bar Plot of W_top_triangular')
plt.gca().invert_yaxis()
plt.show()

plt.barh(y_positions, W_center_triangular)
plt.xlabel('Contribution to synergy')
plt.ylabel('EMG channel')
plt.yticks(y_positions)  # shows 1 through 8 on the y-axis
plt.title('Horizontal Bar Plot of W_top_triangular')
plt.gca().invert_yaxis()
plt.show()

plt.barh(y_positions, W_low_center_triangular)
plt.xlabel('Contribution to synergy')
plt.ylabel('EMG channel')
plt.yticks(y_positions)  # shows 1 through 8 on the y-axis
plt.title('Horizontal Bar Plot of W_top_triangular')
plt.gca().invert_yaxis()
plt.show()



H_1 = np.concatenate([np.full(200, 1), np.linspace(1, 0, 200), np.full(200, 0), np.linspace(0, 1, 200), np.full(200, 1)]).reshape(1, 1000)
H_2_2 = np.concatenate([np.full(200, 0), np.linspace(0, 1, 200), np.full(200, 1), np.linspace(1, 0, 200), np.full(200, 0)]).reshape(1, 1000)
H_2 = np.vstack((H_1, H_2_2)) # 1,2 synergy
H_3_1 = np.concatenate([np.full(76,1), np.linspace(1,0,76), np.full(684,0), np.linspace(0,1,76), np.full(76,1)])
H_3_2 = np.concatenate([np.full(152,0), np.linspace(0,1,76), np.full(76,1), np.linspace(1,0,76), np.full(228,0), np.linspace(0,1,76), np.full(76,1),
                         np.linspace(1,0,76), np.full(152,0)])
H_3_3 = np.concatenate([np.full(380,0), np.linspace(0,1,76), np.full(76,1), np.linspace(1,0,76), np.full(380,0)])
H_3 = np.vstack((H_3_1, H_3_2, H_3_3))

x_1 = np.linspace(0, H_1.shape[1], H_1.shape[1])
x_2 = np.linspace(0, H_2.shape[1], H_2.shape[1])    
x_3 = np.linspace(0, H_3.shape[1], H_3.shape[1])

plt.plot(x_1, H_1.flatten(), color='blue')
plt.xlabel('samples')
plt.ylabel('activation value')
plt.title('Activation in case of 1 synergy')
plt.show()

plt.figure(figsize=(8, 9)) 
plt.subplot(2, 1, 1)
plt.plot(x_2, H_2[0], color='blue')
plt.title('synergy 1')
plt.xlabel("samples")
plt.ylabel("activation value")
plt.subplot(2, 1, 2)
plt.plot(x_2, H_2[1], color='red')
plt.title('synergy 2')
plt.xlabel("samples")
plt.ylabel("activation value")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 9)) 
plt.subplot(3, 1, 1)
plt.plot(x_3, H_3[0], color='blue')
plt.title('synergy 1')
plt.xlabel("samples")
plt.ylabel("activation value")
plt.subplot(3, 1, 2)
plt.plot(x_3, H_3[1], color='red')
plt.title('synergy 2')
plt.xlabel("samples")
plt.ylabel("activation value")
plt.subplot(3, 1, 3)
plt.plot(x_3, H_3[2], color='green')
plt.title('synergy 3')
plt.xlabel("samples")
plt.ylabel("activation value")
plt.tight_layout()
plt.show()
