import matplotlib.pyplot as plt
import numpy as np
import csv

datas=open("./advertising.csv", "r", encoding="utf-8")
datas=csv.reader(datas)

#Batch Gradient Descent
learning_rate=0.0000009
num_iteration=1000000

w_1, w_0=0.5, 0.3

data_list=[]
target_list=[]

i=0

for data in datas:
    if i==0:
        i=1
        continue
    data_list.append(np.array([float(data[0]), 1])) #(200, 2)
    target_list.append(np.array([float(data[-1])])) #(200, 1)


data_list=np.array(data_list) # (200, 2)

max_0_data=max(np.transpose(data_list)[0])
min_0_data=min(np.transpose(data_list)[0])

data_list=np.transpose(data_list) #(2, 200)

plt.scatter(data_list[0], np.transpose(target_list)[0])

data_list[0]=(data_list[0]-min_0_data)/(max_0_data - min_0_data)
data_list=np.transpose(data_list) #(200, 2)

alpha=1/(max_0_data - min_0_data)
beta=-(min_0_data/(max_0_data - min_0_data))


target_list=np.transpose(np.array(target_list)) #(1, 200)

max_target_data=max(target_list[0])
min_target_data=min(target_list[0])



target_list[0] = (target_list[0] - min_target_data) / (max_target_data - min_target_data)
# target_list: (1, 200)

target_list=np.transpose(target_list) #(200, 1)

gamma=1/(max_target_data - min_target_data)
delta= -(min_target_data/(max_target_data - min_target_data))

parameter_list= np.array([[float(w_1)], [float(w_0)]]) # (2, 1)

for i in range(num_iteration):
    predict_list=data_list @ parameter_list #(200, 2) @ (2, 1) -> (200, 1)
    parameter_list[0] -= learning_rate * (np.transpose(data_list)[0] @ (predict_list - target_list)) # (1, 200) @ (200, 1)
    parameter_list[1] -= learning_rate * (np.transpose(data_list)[1] @ (predict_list - target_list))


origin_w1=(parameter_list[0][0] * alpha)/gamma
origin_w0=(parameter_list[0][0] * beta + parameter_list[1][0] - delta) / gamma


total_loss=np.transpose((data_list @ parameter_list) - target_list) @ ((data_list @ parameter_list) - target_list)
print("Total Loss: ",total_loss[0][0])

a=np.linspace(0, 300, 3001)
plt.plot(a, a*origin_w1 + origin_w0, c="r")
plt.show()