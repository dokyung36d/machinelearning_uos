import matplotlib.pyplot as plt
import numpy as np
import csv


datas=open("./advertising.csv", "r", encoding="utf-8")
datas=csv.reader(datas)

#Batch Gradient Descent
learning_rate=0.000008
num_iteration=200 * 12000

w_1, w_0=1, 1
j=0
x_list=[]
y_list=[]

data_list=[]

for data in datas:
    data_list.append(list(data))

data_list = data_list[1:]

for i in range(num_iteration):

    random_index=np.random.randint(0, len(data_list)) #훈련시킬 data를 랜덤하게 선택하는 것이 Batch Gradient Descent와 차이점

    input_data, target_data=float(data_list[random_index][0]), float(data_list[random_index][-1])
    predict = w_1 * input_data + w_0
    
    loss=predict-target_data
    w_1-= learning_rate * loss * input_data
    w_0-= learning_rate * loss
    x_list.append(input_data)
    y_list.append(target_data)

def find_total_loss(w_1, w_0):
    total_loss=0
    for i in range(len(data_list)):
        loss=abs(float(data_list[i][-1])-(float(data_list[i][0]) * float(w_1) + w_0))
        total_loss+=loss
    return total_loss

total_loss=find_total_loss(w_1, w_0)
print("total loss: ",total_loss)
print("w_1: ", w_1)
print("w_0: ", w_0)


plt.scatter(x_list, y_list, c="r")
a=np.linspace(0, 300, 3001)
plt.plot(a, a * w_1 + w_0)
plt.show()