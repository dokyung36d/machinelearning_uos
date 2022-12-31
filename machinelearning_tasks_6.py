import matplotlib.pyplot as plt
import numpy as np
import csv

np.random.seed(10)

lr=0.01
num_iteration= 10000#5000

datas=open("./iris.csv", "r", encoding="utf-8")
datas=csv.reader(datas)
i=0
info=[]
target_list=[]

for data in datas:
    if i==0:
        i=1
        continue
    if list(data)[-1] =="Setosa":
        continue
    info.append(np.array([float(list(data)[0]), float(list(data)[1]), float(list(data)[2]), float(list(data)[3])]))
    if list(data)[-1] == "Versicolor":
        target_list.append(0)
    elif list(data)[-1] == "Virginica":
        target_list.append(1)

info=np.array(info) #(100, 4)
info=np.transpose(info) #(4, 100)

sepal_width_mean=np.mean(info[0])
sepal_width_std=np.std(info[0])

for i in range(4):
    mean=np.mean(info[i])
    std=np.std(info[i])
    info[i] = (info[i] - mean) / std

target_list=np.expand_dims(np.array(target_list), axis=1) #(100, 1)
target_list=np.transpose(target_list) #(1, 100)

class NeuralNet:
    def __init__(self):
        self.w_1=np.random.standard_normal(size= (3, 4))
        self.w_2=np.random.standard_normal(size= (2, 3))
        self.w_3=np.random.standard_normal(size= (1, 2))

        self.b_1=np.random.standard_normal(size= (3, 1))
        self.b_2=np.random.standard_normal(size= (2, 1))
        self.b_3=np.random.standard_normal(size= (1, 1))

    def forward(self, x): # x==info (4, 100)
        z_1= self.w_1 @ x + self.b_1 #(3, 100)
        a_1= 1 / (1 + np.exp( - z_1)) #(3, 100)

        z_2= self.w_2 @ a_1 + self.b_2 #(2, 100)
        a_2= 1 / (1 + np.exp( - z_2)) #(2, 100)

        z_3= self.w_3 @ a_2 + self.b_3 #(1, 100)
        a_3= 1 / (1 + np.exp( - z_3)) #(1, 100)

        return [a_1, a_2, a_3]

    def update(self, target, model_output, x): #backpropagation, model_output = NeuralNet.forward(x)[2]
        a_3=self.forward(x)[2]#(1, 100)
        a_2=self.forward(x)[1] #(2, 100)
        a_1=self.forward(x)[0] #(3, 100)

        delta_3_1 = model_output - target #(1, 100)

        delta_2_1 = delta_3_1 * self.w_3[0][0] * a_2[0] * (1 - a_2[0]) #(1, 100)
        delta_2_2 = delta_3_1 * self.w_3[0][1] * a_2[1] * (1 - a_2[1]) #(1, 100)

        delta_1_1 = (delta_2_1 * self.w_2[0][0] + delta_2_2 * self.w_2[1][0]) * a_1[0] * (1 - a_1[0]) #(1, 100)
        delta_1_2 = (delta_2_1 * self.w_2[0][1] + delta_2_2 * self.w_2[1][1]) * a_1[1] * (1 - a_1[1]) #(1, 100)
        delta_1_3 = (delta_2_1 * self.w_2[0][2] + delta_2_2 * self.w_2[1][2]) * a_1[2] * (1 - a_1[2]) #(1, 100)

        self.w_3 -= (delta_3_1 @ np.transpose(a_2)) / len(info[0])

        self.w_2[0] -= lr * (delta_2_1 @ np.transpose(a_1))[0] / len(info[0])
        self.w_2[1] -= lr * (delta_2_2 @ np.transpose(a_1))[0] / len(info[0])

        self.w_1[0] -= lr * (delta_1_1 @ np.transpose(x))[0] / len(info[0])
        self.w_1[1] -= lr * (delta_1_2 @ np.transpose(x))[0] / len(info[0])
        self.w_1[2] -= lr * (delta_1_3 @ np.transpose(x))[0] / len(info[0])

        self.b_3 -= lr * np.expand_dims(np.mean(delta_3_1, axis=1), axis=-1)

        self.b_2[0][0] -= lr * np.mean(delta_2_1, axis=1)[0]
        self.b_2[1][0] -= lr * np.mean(delta_2_2, axis=1)[0]

        self.b_1[0][0] -= lr * np.mean(delta_1_1, axis=1)[0]
        self.b_1[1][0] -= lr * np.mean(delta_1_2, axis=1)[0]
        self.b_1[2][0] -= lr * np.mean(delta_1_3, axis=1)[0]

def find_loss(model_output, target): #model_output: (1, 100) # target: (1, 100)
    loss= -(target * np.log(model_output) + (1-target) * np.log(1 - model_output))/len(model_output[0])

    return np.mean(loss)

x=NeuralNet()

for _ in range(num_iteration):
    result= x.forward(info)
    x.update(target_list, result[2], info)

print("Total Loss: ", find_loss(result[2], target_list))

plt.scatter(result[1][0][:50], result[1][1][:50], c="r")
plt.scatter(result[1][0][50:], result[1][1][50:], c="b")

a=np.linspace(np.min(result[1][0]), np.max(result[1][0]), 101)
plt.plot(a, -(x.w_3[0][0] * a + x.b_3[0][0])/x.w_3[0][1])
plt.show()