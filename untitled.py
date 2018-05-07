import numpy as np
import matplotlib.pyplot as plt
import random
import math
fig1 = plt.figure()
ax = fig1.add_subplot(111)
x = np.linspace(-10, 10, 500)
y = []
for i in range(0, len(x)):
	y.append((math.erf(x[i])))
y = np.array(y)
print(x.dtype)
y.dtype = 'float'
print(y)
ax.plot(x, y, label = "black")

fig1.show()
input("Press any key to continue")