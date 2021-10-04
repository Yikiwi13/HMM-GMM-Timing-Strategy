import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from class_hmm_gmm_base import *
from class_hmm_xgb_base import *

class TransferMatrixPredictions():
    def __init__(self,path,index,N_stats,emission_matrix_depiction,evaluation_time_span,r_ceiling,r_floor,window,cycle):
        '''

        :param path: 'C:/Users/wangtao/Desktop/0919data' (address in str)
        :param index: 1：上证综指；2：上证50； 3：中证500； 4：深圳成指 (num)
        :param N_stats: number of hidden states (num)
        :param emission_matrix_depiction: GMM/XGB (str）
        :param evaluation_time_span: (num)
        :param r_ceiling: rate of return that marks the ceiling (float)
        :param r_floor: rate of return that marks the floor
        :param window: the length of window when predicting in a roll(int)
        :param cycle: the length of days predicted ahead(int)
        :return: 20160401----20210803 samples with stock_code, trade_date, features and label (dataframe)
        '''

        #Hidden states
        self.N_stats=N_stats

        #Load the data
        filename=os.listdir(path)
        data=pd.read_csv(path+'/'+filename[index-1])
        self.data=data


        #Triple Barrier Method
        self.TripleBarrierLabel(evaluation_time_span,r_ceiling,r_floor)

        #Rolling Predictions
        self.method=emission_matrix_depiction
        self.WindowParallelEstimation(N_stats,window,cycle,start_index=0)

        #Visualization

        self.PlotClassifier()

        if self.N_stats==3:
            self.PlotPredNReal()




    '''
    =========================================
    Triple Barrier Method returns the label
    =========================================
    '''
    def TripleBarrierLabel(self,evaluation_time_span,r_ceiling,r_floor):
        '''
        Definition: this is a posteriori label that depicts the future stock state at current spot within future t-day span
        :param data: dataframe
        :param t: the horizontal days span
        :param ceiling: the rate of return that defines the ceiling
        :param floor: the rate of return that defines the floor
        :return: original dataframe with label
        '''
        label=[]
        for i in range(len(self.data)-evaluation_time_span):
            flag=0
            now=1
            while now<evaluation_time_span:
                if self.data.close[i+now]>=(1+r_ceiling)*self.data.close[i]:
                    flag=1
                    break
                elif self.data.close[i+now]<=(1-r_floor)*self.data.close[i]:
                    flag=-1
                    break
                else:
                    now+=1
            label.append(flag)
        self.data=self.data.iloc[:-evaluation_time_span,:]
        self.data['label']=np.array(label)

        return 0


    '''
    ===============================================================================================================
        #1 Training with GMM_HMM Model (this is a counterpart function of function:WindowParallelEstimation(...))
    ===============================================================================================================
    '''
    def GMM_main(self,train_data,N_stats):
        train_data=np.array(train_data)
        mms=MinMaxScaler()
        train_data=mms.fit_transform(train_data)
        train_data=train_data[:,3:]

        #initializing
        model=HMM_GMM(train_data,N_stats)
        model.train_GMM_HMM(train_data,20) #retrieved the parameters
        print(model)
        prob_max,path=model.Viterbi(train_data)
        last_known_state=path[-1]
        trans_mat=model.A
        trans_mat=np.mat(trans_mat)
        return trans_mat,int(last_known_state)

    '''
      ===============================================================================================================
          #2 Training with XGB_HMM Model (this is a counterpart function of function:WindowParallelEstimation(...))
      ===============================================================================================================
    '''

    def XGB_main(self,train_data,N_stats):
        train_data=train_data[:,3:]
        model=dict()
        model=HMM_XGB(train_data,N_stats)
        #training the model
        model.train_HMM_XGB(train_data,20)
        prob_max,path=model.Viterbi(train_data)
        trans_mat=model.A
        trans_mat=np.mat(trans_mat) #转换为矩阵以后，可以直接通过矩阵M**2实现矩阵的二次方
        last_known_state=path[-1]
        return trans_mat,int(last_known_state)


    '''
    =================================================================================================================
        Rolling Predictions——①trained with observation sequences,②predicted with transfer matrix only
    =================================================================================================================
    '''

    def WindowParallelEstimation(self,N_stats,window,cycle,start_index=0):
        '''

        :param window: the length of data for training parameters(int)
        :param cycle: the time-span of forecasting and the cyclic span of parameter-rotation(int)
        :param start_index: the current window's starting point's index
        :return: the merged result of rolling predictions
        '''



        label=np.array(self.data['label'])
        price=np.array(self.data.iloc[:,2])
        obs=self.data.iloc[:,3:-1]
        predicted=[]     #prediction list
        while start_index+window<len(self.data):
            #就一直移动
            train_data=obs.iloc[start_index:start_index+window,:]
            train_data=np.array(train_data)



            if self.method=='GMM':
                trans_mat,last_known_state=self.GMM_main(train_data,N_stats)
            elif self.method=='XGB':
                trans_mat,last_known_state=self.XGB_main(train_data,N_stats)
            else:
                print('The emission matrix is not supported to be depicted by this method, you are welcomed to choose: XGB or GMM')


            #一次预测
            for i in range(1,cycle+1):
                temp=np.array(trans_mat**i)
                newly_pred=np.argmax(temp[last_known_state])
                predicted.append(newly_pred)
            start_index+=cycle#移动窗口
        #获取待预测数据的长度：
        pred_length=len(self.data)-window
        predicted=predicted[:pred_length]
        label=label[window:]
        price=price[window:]
        self.predicted=np.array(predicted)
        print(len(self.predicted))
        self.label=np.array(label)
        print(len(self.label))
        self.price=np.array(price)
        print(len(self.price))
        return 0



    '''
       =================================================================================================================
                                    Visualization [plain+plus version(only available when N_stats==3)]
       =================================================================================================================
       '''

        #plain visualization
    def PlotClassifier(self):
        color_pool=['navy','lightpink','maroon','yellow','green','purple','orange']
        color=color_pool[:self.N_stats]
        for i in range(len(self.predicted)):
            for j in range(self.N_stats):
                if self.predicted[i]==j:
                    plt.scatter(i,self.price[i],c=color[j])
                else:
                    continue

        plt.title(self.method+'_HMM Transfer Matrix Predictions',fontdict={'family':'serif',
                                                                            'weight':'normal',
                                                                            'color':'black',
                                                                            'size':16})
        plt.grid()
        plt.show()
        return 0

        #2 only used when N_stats=3 so that labels and classifier-number returned by unsupervised classification can correspond to each other
    def Cluster(self,marks):
        #cluster predictions into the number of types of marks
        '''

        :param marks: 0/1/2
        :return:
        '''
        cluster_x=dict()
        cluster_y=dict()

        for i in np.sort(np.unique(marks)):
            cluster_x[i]=[]
            cluster_y[i]=[]
        for j in range(len(marks)):
            for i in np.sort(np.unique(marks)):
                if marks[j]==i:
                    cluster_x[i].append(j)
                    cluster_y[i].append(self.price[j])
                else:
                    continue
        return cluster_x,cluster_y

    def MatchScore(self,cond_index,condition_dict,predicted,temp):
        loose_count=0
        strict_count=0
        for i in range(len(predicted)):
            if np.abs(condition_dict[predicted[i]]-temp[i])<=1:
                loose_count+=1
                if condition_dict[predicted[i]]==temp[i]:
                    strict_count+=1
            else:
                continue
        loose_score=loose_count/len(predicted)
        strict_score=strict_count/len(predicted)
        print("loose score%d:%.4f"%(cond_index,loose_score))
        print("strict score%d:%.4f"%(cond_index,strict_score))
        return strict_score

    def MatchClusterWithLabel(self):
        self.cluster_x,self.cluster_y=self.Cluster(self.predicted)
        predicted=self.predicted
        label=self.label
        temp=label+1

        condition=dict()
        for i in range(6):
            condition[i]=dict()

        condition[0]=dict({0:0,1:1,2:2})
        condition[1]=dict({0:0,1:2,2:1})
        condition[2]=dict({0:1,1:0,2:2})
        condition[3]=dict({0:1,1:2,2:0})
        condition[4]=dict({0:2,1:0,2:1})
        condition[5]=dict({0:2,1:1,2:0})
        score=dict()
        for i in range(6):
            score[i]=self.MatchScore(i+1,condition[i],predicted,temp)
        z=[]
        for j in score.values():
            z.append(j)
        acc=np.max(z)
        print('acc is %.4f'%acc)

        best_index=np.argmax(z)+1 #locate the index where the predicting accuracy ranks the highest
        according={1:['y','c','m'],2:['y','m','c'],3:['c','y','m'],4:['c','m','y'],5:['m','y','c'],6:['m','c','y']}
        matchcolor=according[best_index]
        self.matchcolor=matchcolor



    def PlotPredNReal(self):
        self.MatchClusterWithLabel()

        fig = plt.figure(figsize = (8,5))
        #subplot1 预测状态与股价图
        ax = fig.add_subplot(211)
        type1 = ax.scatter(self.cluster_x[0], self.cluster_y[0],  c = self.matchcolor[0])
        type2 = ax.scatter(self.cluster_x[1], self.cluster_y[1],  c = self.matchcolor[1])
        type3 = ax.scatter(self.cluster_x[2], self.cluster_y[2],  c = self.matchcolor[2])

        plt.xlabel('time')
        plt.ylabel('stock price')
        plt.grid()
        plt.title('Predictions with Transfer-Matrix Rolling Window',fontdict={'family':'serif',
                                                                                 'weight':'normal',
                                                                                 'color':'black',
                                                                                 'size':16})
        ax.legend((type1, type2, type3), ("state#1", "state#2", "state#3"), loc = 0)

        #subplot2 triple barrier标记点与股价图
        ax = fig.add_subplot(212)
        label=self.label+1
        cluster_x,cluster_y=self.Cluster(label)
        type1 = ax.scatter(cluster_x[0], cluster_y[0],  c = 'y')
        type2 = ax.scatter(cluster_x[1], cluster_y[1],  c = 'c')
        type3 = ax.scatter(cluster_x[2], cluster_y[2],  c = 'm')

        plt.xlabel('time')
        plt.ylabel('stock price')
        plt.grid()
        plt.title('Triple Barrier Marks',fontdict={'family':'serif',
                                                   'weight':'normal',
                                                   'color':'black',
                                                   'size':16})
        ax.legend((type1, type2, type3), ("going down", "oscillation", "rising up"), loc = 0)


        plt.tight_layout()
        plt.show()






TransferMatrixPredictions('C:/Users/wangtao/Desktop/0919data',3,3,'GMM',5,0.02,0.01,100,5)
