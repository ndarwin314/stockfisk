import numpy as np
import matplotlib.pyplot as plt

x = np.load("x_arr_3820.npy")
y = np.load("y_arr_3820.npy")
filter = np.array([1.0,2,3,2,1])
filter /= filter.sum()

c1 = np.convolve(y[0], filter, mode="same")
c2 = np.convolve(y[1], filter, mode="same")
c3 = np.convolve(y[2], filter, mode="same")


plt.xlim(0, 2740)
plt.ylim(0, 1)
plt.xlabel("Epochs")
plt.ylabel("Win %")

plt.plot(x, y[0], label="Random")
plt.plot(x, y[1], label="Max BP")
plt.plot(x, y[2], label="Heuristic")
plt.grid()
plt.legend()
plt.savefig("test.png")
print(x[np.argmax(y[2])])
