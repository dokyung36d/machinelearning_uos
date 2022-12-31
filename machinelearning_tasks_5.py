import matplotlib.pyplot as plt
import numpy as np
import csv


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
    info.append(np.array([float(list(data)[1]), float(list(data)[2]), 1]))
    if list(data)[-1] == "Versicolor":
        target_list.append(0)
    elif list(data)[-1] == "Virginica":
        target_list.append(1)


num_iteration=200000
learning_rate=0.01


target_list=np.array(target_list)

info=np.array(info) #(100, 3)
info=np.transpose(info) #(3, 100)

plt.scatter(info[0][0:50], info[1][0:50], c="r")
plt.scatter(info[0][50:100], info[1][50:100], c="b")

sepal_width_mean=np.mean(info[0])
sepal_width_std=np.std(info[0])

info[0]= (info[0] - sepal_width_mean) / sepal_width_std

petal_length_mean= np.mean(info[1])
petal_length_std= np.std(info[1])

info[1]= (info[1] - petal_length_mean) / petal_length_std

w1, w2, w0= 0.1, 0.5, 1 #0.1,0.5,1

for i in range(num_iteration):
    w=np.array([w1, w2, w0])

    predict=w @ info #(100, )
    predict_sigmoid= 1 / (1 + pow(np.e, -predict))

    loss= -(target_list * np.log(predict_sigmoid) + (1-target_list) * np.log(1-predict_sigmoid)) / len(info[0])

    if i==0:
        print("Before Training: ", abs(np.sum(loss)))

    w1 -= learning_rate * float(-(np.sum(info[0] * (target_list - predict_sigmoid)) / len(info[0])))
    w2 -= learning_rate * float(-(np.sum(info[1] * (target_list - predict_sigmoid)) / len(info[0])))
    w0 -= learning_rate * float(-(np.sum((target_list - predict_sigmoid)) / len(info[0])))

print("After Training: ", abs(np.sum(loss)))

x1=np.linspace(2, 3.75, 100)
x1_scailing = (x1 - sepal_width_mean) / sepal_width_std
x2_scailing = -((w1*x1_scailing + w0) / w2)
x2 = petal_length_std * x2_scailing + petal_length_mean
plt.plot(x1, x2, c="g")

plt.show()
