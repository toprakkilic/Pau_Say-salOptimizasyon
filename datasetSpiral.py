import numpy as np
import math

B = 4.2
N = 300
Tall = np.array([]).reshape(2, -1)
for i in range(0,int(N/2)):
    theta = math.pi/2 + (i-1)*((2*B-1)/N)*math.pi
    A = np.array([theta*math.cos(theta), theta*math.sin(theta)]) .reshape(2, 1)
    Tall = np.hstack((Tall,A))

Tall = np.hstack((Tall,-Tall))
Tmax = math.pi/2 + ((N/2-1)*(2*B-1)/N)*math.pi
ti = Tall.transpose()/Tmax
yi = np.hstack((np.ones(int(N/2)),-np.ones(int(N/2))))

#-------------------------------------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
plt.scatter(ti[:int(N/2),0], ti[:int(N/2),1], color='g',s=5,marker='o',alpha=0.99)
plt.scatter(ti[int(N/2):,0], ti[int(N/2):,1], color='r',s=5,marker='x',alpha=0.99)
plt.xlabel('$t_1$')
plt.ylabel('$t_2$')
plt.title('Dataset Spiral', fontstyle='italic')
plt.grid(color='green',linestyle='--',linewidth=0.1)
ax = plt.gca()
ax.xaxis.set_tick_params(labelbottom=False)
ax.yaxis.set_tick_params(labelleft=False)
plt.show()
