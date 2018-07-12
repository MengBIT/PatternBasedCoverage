# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 16:24:28 2018

@author: XMeng
"""
from numpy import *
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import itertools

def map_matrix(Grid,Pstart_x,Pstart_y):
    chessboardTemp = np.zeros((Grid,Grid))
    chessboardTemp=chessboardTemp.astype(int)
    chessboardTemp[0,:]=1
    chessboardTemp[Grid-1,:]=1
    chessboardTemp[:,0]=1
    chessboardTemp[:,Grid-1]=1
    for i in range(Robot_NUM):
        chessboardTemp[int(Pstart_x[i]),int(Pstart_y[i])]=1
    '''
    bulidings=[{ "x": 1, "y": 2, "l": 3, "w": 1},
               { "x": 5, "y": 2, "l": 2, "w": 2}]
    #print(bulidings)
    for item in bulidings:
        for x in range(item["x"],item["x"]+item["l"]):
            for y in range(item["y"],item["y"]+item["w"]):
                    chessboardTemp[x][y] = 1
                
    [x,y,l,w]=[item["x"],item["y"],item["l"],item["w"]]
    point=[]
    for item in bulidings:
        point.append([x-1,y-1])
        point.append([x-1,y+w])
        point.append([x+l,y-1])
        point.append([x+l,y+w])
        #print(chessboardTemp)
    '''
    return chessboardTemp

'''
COUNT=0
allmodel=[]
def perm(n,begin,end):
    global COUNT
    global allmodel
    if begin>=end:
        allmodel.append(n)
        COUNT+=1
    else:
        i=begin
        for num in range(begin,end):
            n[num],n[i]=n[i],n[num]
            perm(n,begin+1,end)
            n[num],n[i]=n[i],n[num]
    return allmodel
'''

def Qifa(Point_x,Point_y,G,fitness,model):
    global grid
    m=grid-2
    index=0
    Point_x=int(Point_x)
    Point_y=int(Point_y)
    if (G[Point_x+1,Point_y]+G[Point_x,Point_y+1]+G[Point_x-1,Point_y]+G[Point_x,Point_y-1]==4):
        index=1;
        next_Point_x=Point_x
        next_Point_y=Point_y
        next_fitness=fitness
        G_next=G
    elif (G[Point_x+1,Point_y]+G[Point_x,Point_y+1]+G[Point_x-1,Point_y]+G[Point_x,Point_y-1]!=4):
        n=[1,2,3,4]
        #allmodel=perm(n,0,len(n))
        allmodel=list(itertools.permutations(n,4)) #生成全排列组合
        model=int(model)
        #print(model)  ##调试用
        choosefrom=allmodel[model]
        for i in range(4):
            choose=choosefrom[i]
            if (choose==1 and G[Point_x,Point_y+1]==0):
                next_Point_x=Point_x
                next_Point_y=Point_y+1
                next_fitness=fitness+1
                G[Point_x,Point_y+1]=1
                break
            elif (choose==2 and G[Point_x+1,Point_y]==0):
                next_Point_x=Point_x+1
                next_Point_y=Point_y
                next_fitness=fitness+1
                G[Point_x+1,Point_y]=1
                break
            elif (choose==3 and G[Point_x,Point_y-1]==0):
                next_Point_x=Point_x
                next_Point_y=Point_y-1
                next_fitness=fitness+1
                G[Point_x,Point_y-1]=1
                break
            elif (choose==4 and G[Point_x-1,Point_y]==0):
                next_Point_x=Point_x-1
                next_Point_y=Point_y
                next_fitness=fitness+1
                G[Point_x-1,Point_y]=1
                break
        G_next=G
    '''
    elif (G[Point_x+1,Point_y]+G[Point_x,Point_y+1]+G[Point_x-1,Point_y]+G[Point_x,Point_y-1]==4 and fitness==m*m):
        index=1;
        next_Point_x=Point_x
        next_Point_y=Point_y
        next_fitness=fitness
        G_next=G
    '''
    return next_Point_x,next_Point_y,G_next,next_fitness,index

def GA_search(obs_NUM,obs_fit,Pstart_x,Pstart_y,Robot_NUM,model,plotting):
    global grid
    global Gm
    global Np
    global model_NUM
    D=Robot_NUM
    CR=0.9
    MR=0.01
    #可选择的模式数量
    Fangan=np.zeros((Np,model_NUM*2,D))
    Fangan_model=np.random.randint(0,model-1,size=[Np,model_NUM,D])
    Fangan_step=np.random.randint(1,(grid-2)*(grid-2),size=[Np,model_NUM,D])
    Fangan[:,np.s_[::2],:]=Fangan_model
    Fangan[:,np.s_[1::2],:]=Fangan_step
    Fangan=Fangan.astype(int)
    Gmax=np.zeros(Gm)
    Fangan_final=np.zeros((Gm,model_NUM*2,D))
    fitness=np.zeros(Np)
    fitness_choose=np.zeros(Np)
    #先对初始解计算适应值
    for i in range(Np):
        Env1=map_matrix(grid,Pstart_x,Pstart_y)
        fitness[i]=search(Env1,D,Pstart_x,Pstart_y,Fangan[i,:,:],plotting)
    #开始循环寻优
    for G in range(Gm):
        #交叉
        Fangan_next_1=Fangan
        #print('Fangan_next_1:\n',Fangan_next_1)  ##调试用
        for ii in range(Np):
            if random.random()<CR:
                #dx=np.arange(0,Np-1,4) 
                dx=random.sample(range(0,Np-1),3)
                dx_pos=np.where(dx==ii)
                A1=np.delete(dx,dx_pos)
                index_x1=np.random.randint(0,D-1)
                temp_x=Fangan_next_1[A1[0],:,index_x1]
                Fangan_next_1[A1[0],:,index_x1]=Fangan_next_1[A1[1],:,index_x1]
                Fangan_next_1[A1[1],:,index_x1]=temp_x 
        Fangan_next_2=Fangan_next_1
        #print('Fangan_next_2--1:\n',Fangan_next_2)   ##调试用
        #变异
        for ii in range(Np):
            if random.random()<MR:
                index_x2=np.random.randint(0,D-1)
                Fangan_again=np.zeros(model_NUM*2)
                Fangan_model_again=np.random.randint(0,model-1,model_NUM)
                Fangan_step_again=np.random.randint(1,(grid-2)*(grid-2),model_NUM)
                Fangan_again[np.s_[::2]]=Fangan_model_again
                Fangan_again[np.s_[1::2]]=Fangan_step_again
                Fangan_next_2[ii,:,index_x2]=Fangan_again
        #print('Fangan_next_2--2:\n',Fangan_next_2)    ##调试用
        #选择
        for i in range(Np):
            #print('---------------Env-----------\n',Env)
            Env2=map_matrix(grid,Pstart_x,Pstart_y)
            fitness_choose[i]=search(Env2,D,Pstart_x,Pstart_y,Fangan_next_2[i,:,:],plotting)
        compare=np.array(fitness_choose>fitness)+0
        #print(compare)  ##调试用
        compare_reshape=compare.reshape(Np,1,1)
        #print(compare_reshape)   ##调试用
        Fangan_next_3=np.multiply(compare_reshape,Fangan_next_2)+np.multiply((1-compare_reshape),Fangan)
        fitness_final=np.multiply(compare,fitness_choose)+np.multiply((1-compare),fitness)
        value_max=fitness_final.max()
        Gmax[G]=value_max
        pos_max=np.argmax(fitness_final)           
        Fangan_final[G,:,:]=Fangan_next_3[pos_max,:,:]
        #保存最优个体
        fitness=fitness_final
        Fangan=Fangan_next_3
    best_pos=np.argmax(Gmax)
    Fangan_final_choose=Fangan_final[best_pos,:,:]
    return Fangan_final_choose
  
      
def search(Env,Robot_NUM,Pstart_x,Pstart_y,Fangan,plotting):
    global grid
    global model_NUM
    #起始点初始值
    Point_x=Pstart_x
    Point_y=Pstart_y
    
    fitness=Robot_NUM
    All_grid=(grid-2)*(grid-2)
    next_Point_x=np.zeros(Robot_NUM)
    next_Point_y=np.zeros(Robot_NUM)
    index=np.zeros(Robot_NUM)
    Robot_model=np.zeros(Robot_NUM)
    Robot_model=Robot_model.astype(int)
    Robot_step=np.zeros(Robot_NUM)
    model=Fangan[0,:]
    step=Fangan[1,:]
    time=0
    coverage_NUM=fitness
    for i in range(All_grid-Robot_NUM):
        for j in range(Robot_NUM):
            #print('before_fitness:',fitness)
            next_Point_x[j],next_Point_y[j],Env_next,next_fitness,index[j]=Qifa(Point_x[j],Point_y[j],Env,fitness,model[j])
            #print('after_fitness:',next_fitness)
            #print('######################################################################')
            #print(Env_next)
            #print('######################################################################')
            Robot_step[j]=Robot_step[j]+1
            if (Robot_step[j]==step[j] and Robot_model[j]<=model_NUM*2-1-2):
                Robot_step[j]=0
                Robot_model[j]=Robot_model[j]+2
                Robot_model[j]=int(Robot_model[j])
                model[j]=Fangan[Robot_model[j],j]
            fitness=next_fitness
            #print('search nei fitness1',fitness)
            Env=Env_next
        coverage_NUM=fitness
        time+=1
        Point_x=next_Point_x
        Point_y=next_Point_y
        #print('search nei fitness',coverage_NUM)
        if index.sum()==Robot_NUM:
            break
        elif coverage_NUM==All_grid:
            break
            
    print('for wai fitness:',coverage_NUM)
    Fugailv=coverage_NUM/All_grid
    Time_cost=(All_grid-time+1)/(All_grid-All_grid/Robot_NUM+1)
    result=Fugailv+Time_cost
    if plotting==1:
        if coverage_NUM==All_grid:
           print('successed!')
        print('覆盖率=',Fugailv*100)
        print('覆盖个数=',coverage_NUM)
        print('最优覆盖个数=',All_grid)
        print('搜索时间=',time-1)
        print('最优搜索时间=',All_grid/Robot_NUM-1)
        print(Env)
        drawShanGe(Env)
    return result

def initial(Robot_NUM,grid):
    Pstart_x=np.zeros(Robot_NUM)
    Pstart_y=np.zeros(Robot_NUM)
    Pstart_x_choosefrom1=np.ones(Robot_NUM)*1
    Pstart_x_choosefrom2=np.ones(Robot_NUM)*(grid-2)
    Pstart_y_choosefrom1=np.ones(Robot_NUM)*1
    Pstart_y_choosefrom2=np.ones(Robot_NUM)*(grid-2)
    suiji_x=np.array(np.random.random(Robot_NUM)>0.5)+0
    Pstart_x1=np.multiply(suiji_x,Pstart_x_choosefrom1)+np.multiply((1-suiji_x),Pstart_x_choosefrom2)
    Pstart_y1=np.random.randint(1,grid-2,Robot_NUM)
    suiji_y=np.array(np.random.random(Robot_NUM)>0.5)+0
    Pstart_y2=np.multiply(suiji_y,Pstart_y_choosefrom1)+np.multiply((1-suiji_y),Pstart_y_choosefrom2)
    Pstart_x2=np.random.randint(1,grid-2,Robot_NUM)
    suiji_xy=np.array(np.random.random(Robot_NUM)>0.5)+0
    Pstart_x=np.multiply(suiji_xy,Pstart_x1)+np.multiply((1-suiji_xy),Pstart_x2)
    Pstart_y=np.multiply(suiji_xy,Pstart_y1)+np.multiply((1-suiji_xy),Pstart_y2)
    Pstart_x=Pstart_x.astype(int)
    Pstart_y=Pstart_y.astype(int)
    return Pstart_x,Pstart_y

def drawShanGe(Env):
    imshow(Env,interpolation='nearest', cmap='bone', origin='lower')
    colorbar(shrink=.92)
    show()



#----------------main----------------------------
if __name__ == '__main__':
    Gm=2
    Np=10
    grid=30
    Robot_NUM=4
    model_NUM=2
    model=24
    NUM=4
    obs_NUM=2
    obs_fit=4
    Pstart_x,Pstart_y=initial(Robot_NUM,grid)  
    plotting=0
    Fangan_final_choose=GA_search(obs_NUM,obs_fit,Pstart_x,Pstart_y,Robot_NUM,model,plotting)
    print('Fangan_final_choose:\n',Fangan_final_choose)
    print('---------------最终结果------------------：')
    plotting=1
    #print(Env)
    #print(Pstart_x)
    Env=map_matrix(grid,Pstart_x,Pstart_y)
    Fugailv=search(Env,Robot_NUM,Pstart_x,Pstart_y,Fangan_final_choose,plotting)
    
    