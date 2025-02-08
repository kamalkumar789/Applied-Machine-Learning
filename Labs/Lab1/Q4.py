import numpy as np
import math as mp
import matplotlib.pyplot as plt

array = np.linspace(0, 2 * mp.pi, num=100)

sin_arr = np.sin(array)

cos_arr = np.cos(array)


plt.figure(figsize=(8, 5))
plt.plot(array, sin_arr, label='sin(x)', linewidth=2)
plt.plot(array, cos_arr, label='cos(x)', linewidth=2, linestyle='dashed')

plt.xlabel('x (radians)')
plt.ylabel('Function value')
plt.title('Sine and Cosine Functions')
plt.legend()
plt.grid(True)

plt.show()