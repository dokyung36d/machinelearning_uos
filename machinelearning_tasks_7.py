import matplotlib.pyplot as plt
import numpy as np
import csv

np.random.seed(10)

lr=0.0001
num_iteration= 5000#5000

datas=open("./iris.csv", "r", encoding="utf-8")
datas=csv.reader(datas)
i=0
j=0
info=[]
target_list=np.zeros((150, 3))

for data in datas:
    if i==0:
        i=1
        continue
    info.append(np.array([float(list(data)[0]), float(list(data)[1]), float(list(data)[2]), float(list(data)[3])]))
    if list(data)[-1] =="Setosa":
        target_list[j][0] =1
    if list(data)[-1] == "Versicolor":
        target_list[j][1] =1
    elif list(data)[-1] == "Virginica":
        target_list[j][2] =1
    
    j+=1 #target_list index 관련

# target list: (150, 3) ->추후 transpose할 수도 있음

info = np.array(info) #(150, 4)
info = np.transpose(info) #(4, 150)

for i in range(4):
    mean=np.mean(info[i])
    std=np.std(info[i])
    info[i] = (info[i] - mean) / std

def ReLu(x, n, m): #n: 행의 갯수, m: 열의 갯수
    for i in range(n):
        for j in range(m):
            x[i][j]=max(0, x[i][j])
    return x

def ReLu_derivative(x, n, m):
    for i in range(n):
        for j in range(m):
            if x[i][j]>0:
                x[i][j]=1
            else:
                x[i][j]=0

    return x

def softmax(x): #input: (3, 150) -> output: (3, 150)
    x_exp=pow(np.e, x) #(3, 150)
    x_exp_sum=x_exp.sum(axis=0) 
    
    return x_exp/x_exp_sum
    

class NeuralNet:
    def __init__(self): #( , )에서 앞쪽이 해당 layer를 통과하였을 때 feature갯수, 뒷부분은 해당 layer를 통과하기 이전의 feature갯수
        self.w_1=np.random.standard_normal(size= (7, 4))
        self.w_2=np.random.standard_normal(size= (8, 7))
        self.w_3=np.random.standard_normal(size= (5, 8))
        self.w_4=np.random.standard_normal(size= (6, 5)) #0    1    2    3    4    5
        self.w_5=np.random.standard_normal(size= (3, 6)) #4 -> 7 -> 8 -> 5 -> 6 -> 3

        self.b_1=np.random.standard_normal(size= (7, 1))
        self.b_2=np.random.standard_normal(size= (8, 1))
        self.b_3=np.random.standard_normal(size= (5, 1))
        self.b_4=np.random.standard_normal(size= (6, 1))
        self.b_5=np.random.standard_normal(size= (3, 1))

    def forward(self, x): # x==info (4, 150)
        z_1= self.w_1 @ x + self.b_1 #(7, 150)
        a_1= ReLu(z_1, 7, 150)

        z_2= self.w_2 @ a_1 + self.b_2  #(8, 150)
        a_2= ReLu(z_2, 8, 150)

        z_3= self.w_3 @ a_2 + self.b_3  #(5, 150)
        a_3= ReLu(z_3, 5, 150)

        z_4= self.w_4 @ a_3 + self.b_4  #(6, 150)
        a_4= ReLu(z_4, 6, 150)

        z_5= self.w_5 @ a_4 + self.b_5  #(3, 150)
        a_5= softmax(z_5)    #(3, 150)

        return [[a_1, a_2, a_3, a_4, a_5], [z_1, z_2, z_3, z_4, z_5]]
        
    def update(self, x): # x==info (4, 150)
        a=self.forward(x)
        a_1, a_2, a_3, a_4, a_5= a[0][0], a[0][1], a[0][2], a[0][3], a[0][4]
        z_1, z_2, z_3, z_4, z_5= a[1][0], a[1][1], a[1][2], a[1][3], a[1][4]

        delta_5= np.transpose(a_5) - target_list #(150, 3) - (150, 3) = (150, 3)
        delta_4= (delta_5 @ self.w_5) * np.transpose(ReLu_derivative(a_4, 6, 150)) #(150, 6)
        delta_3= (delta_4 @ self.w_4) * np.transpose(ReLu_derivative(a_3, 5, 150)) #(150, 5)
        delta_2= (delta_3 @ self.w_3) * np.transpose(ReLu_derivative(a_2, 8, 150)) #(150, 8)
        delta_1= (delta_2 @ self.w_2) * np.transpose(ReLu_derivative(a_1, 7, 150)) #(150, 7)

        self.w_5 -= lr * np.transpose((z_4 @ delta_5) / len(delta_5))  #(3, 6) z_4: (6, 150)
        self.w_4 -= lr * np.transpose((z_3 @ delta_4) / len(delta_5))  #(6, 5)
        self.w_3 -= lr * np.transpose((z_2 @ delta_3) / len(delta_5))  #(5, 8)
        self.w_2 -= lr * np.transpose((z_1 @ delta_2) / len(delta_5))  #(8, 7)
        self.w_1 -= lr * np.transpose((x @ delta_1) / len(delta_5))    #(7, 4)

        self.b_5 -= lr * np.expand_dims(delta_5.sum(axis=0) / len(delta_5), axis=-1) #(3, 1)
        self.b_4 -= lr * np.expand_dims(delta_4.sum(axis=0) / len(delta_5), axis=-1) #(6, 1)
        self.b_3 -= lr * np.expand_dims(delta_3.sum(axis=0) / len(delta_5), axis=-1) #(5, 1)
        self.b_2 -= lr * np.expand_dims(delta_2.sum(axis=0) / len(delta_5), axis=-1) #(8, 1)
        self.b_1 -= lr * np.expand_dims(delta_1.sum(axis=0) / len(delta_5), axis=-1) #(7, 1)

def find_total_loss(model_output): # target list: (150, 3), model_output: (3, 150)
    total_loss= np.transpose(target_list) * (-np.log(model_output)) #(3, 150)
    total_loss = total_loss.sum(axis=-1)
    total_loss = total_loss.sum(axis=-1) / len(target_list)
    
    return total_loss

x=NeuralNet()
graph_x=[]
graph_y=[]

for i in range(1, num_iteration+1):
    result= x.forward(info)
    x.update(info)
    graph_x.append(i)
    graph_y.append(find_total_loss(result[0][-1]))

plt.plot(graph_x, graph_y)
plt.show()