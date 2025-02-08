

import numpy as np

array = np.random.randint(1,101, size = (10,10))

single_array = array.flatten()

for i in range(1, single_array[0]+1):
    if single_array[i] > 50:
        single_array[i] = -1

print(single_array.reshape(10,10))
