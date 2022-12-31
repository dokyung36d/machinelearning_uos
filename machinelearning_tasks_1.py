import matplotlib.pyplot as plt
import numpy as np

position=np.random.uniform(low=-10, high=10, size=(2, 30))

random_line_slope=np.random.uniform(low=-1, high=1)
random_line_noise=np.random.uniform(low=-1, high=1)

red_list=[0]*30

for i in range(30):
    if random_line_slope*position[0][i]+random_line_noise<position[1][i]:
        red_list[i]=1

for i in range(30):
    if red_list[i]==0:
        plt.scatter(position[0][i], position[1][i], c="r", marker="o")
    else:
        plt.scatter(position[0][i], position[1][i], c="b", marker="*")

a=np.linspace(-10, 10, 1001)
plt.plot(a, random_line_slope*a+random_line_noise, c="g", linewidth=2)
plt.show()