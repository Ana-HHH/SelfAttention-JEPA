import numpy as np
from atencionauto import selfAttention 
import seaborn as sns
import matplotlib.pyplot as plt
##Datos aleatorios asignados a un enunciado de 4 palabras

lenght, dim_k, dim_v = 4, 8, 8
Q = np.random.randn(lenght, dim_k)
K = np.random.randn(lenght, dim_k)
V = np.random.randn(lenght, dim_v)


testA = selfAttention(Q, K, V)

atencion, nuevo_v = testA.calculateSelfA()

sns.set()
heatmap = sns.heatmap(atencion, linewidths=1, cmap='YlGnBu', annot=True)
plt.show()
