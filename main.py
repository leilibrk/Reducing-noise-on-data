import math
import matplotlib.pyplot as plt
import numpy as np

data = np.load('btc_price.npy')
n = data.size
D = np.zeros((n - 1, n))
I = np.zeros((n, n))

for i in range(0, n):
    I[i][i] = 1

for i in range(0, n - 1):
    D[i][i] = 1
    D[i][i + 1] = -1

mu = 9999  # lambda
mD = math.sqrt(mu) * D
A = np.vstack((I, mD))

z = np.zeros((n - 1, 1))
y = data.reshape((n, 1))
b = np.vstack((y, z))
# Ax = b
# AtAx=Atb -> x = (AtA)^-1Atb
At = A.transpose()
AtA = np.dot(At, A)
Atb = np.dot(At, b)
AtA_inv = np.linalg.inv(AtA)
x = np.dot(AtA_inv, Atb)
fig = plt.figure()
txt = 'lambda = ' + str(mu)
fig.suptitle(txt, fontsize=14, fontweight='bold')
plt.plot(data, linewidth=3, label='noisy', alpha=0.4, color="blue")
plt.plot(x, 'r', linewidth=3, label='denoised')
plt.legend()
plt.show()
