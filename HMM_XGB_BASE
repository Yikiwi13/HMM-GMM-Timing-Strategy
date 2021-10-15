import xgboost as xgb
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


'''
==================================
    The base class:  HMM CLASS
==================================
'''

class HMM():
    def __init__(self,observations,N_stats):
        '''
        :param observations: observation sequence
        :param self: information of the model that passes through the whole process containing components λ(π,A,B) of Hidden-Markov Model
        :param π: prior probability of state distribution(pi)
        :param A: state transition matrix
        :param N_stats:the number of hidden states preset
        '''
        self.o=observations

        self.N_stats=N_stats

        self.A=np.random.random_sample(size=(N_stats,N_stats))
        self.A=self.A/np.sum(self.A,axis=1)

        self.pi=np.random.random_sample(N_stats)
        self.pi=self.pi/float(np.sum(self.pi))

    '''
   ========================================================================================================================================
                                                             #1 Evaluation
   Evaluating the degree of fitting from model to observation sequences by depicting with: P(O|λ,S)
   The observation sequence should be renewed data each time evaluating, so please mind that the observation is not the self.observations 
   =========================================================================================================================================
   '''
    '''(I) Forward Algorithm'''
    def cal_alpha(self,observations):
        o=observations
        N_samples=np.shape(o)[0]
        N_stats=np.shape(self.pi)[0]
        #initialization
        alpha=np.zeros([N_samples,N_stats])
        alpha[0]=self.pi*self.B[0]
        #iteration
        for t in range(1,N_samples):
            temp=np.dot(alpha[t-1],self.A)
            alpha[t]=temp*self.B[t]
        #np.dot automatically accomodates narray[0] to be (N,1)or(1,N)
        return alpha


    def forward(self,observations):
        o=observations
        alpha=self.cal_alpha(o)
        prob_seq_f=np.log(np.sum(alpha[-1]))
        return prob_seq_f

    '''(II) Backward Algorithm'''

    def cal_beta(self,observations):
        o=observations
        N_samples=np.shape(o)[0]
        N_stats=np.shape(self.pi)[0]
        beta=np.zeros([N_samples,N_stats])
        beta[-1]=1
        for t in range(N_samples-2,-1,-1):
            beta[t]=np.dot(beta[t+1],self.A.T)*self.B[t+1]
        return beta

    def backward(self,observations):
        o=observations
        beta=self.cal_beta(o)
        prob_seq_b=np.dot(beta[0]*self.B[0],self.pi)
        prob_seq_b=np.log(prob_seq_b)
        return prob_seq_b

    '''
    ===========================================================================================================================================================================
                                                                   #2 Decoding
    Traverse first in sequential order to record the maximum probability along certain path that suits with the observation sequence till this state at this time 
    and then traverse in reverse order to fill in the path from the state at t=T with the maximum probability among all the hidden states at time T. 
    *delta: (N_samples,N_stats) store the probabilities
    *psi: (N_samples, N_stats) store the most likely  hidden state  that current state at t is generated from at t-1
    *path: (N_samples,1)
    ===========================================================================================================================================================================
    '''

    '''Viterbi Algorithm'''
    def Viterbi(self,test):
        o=test
        N_samples=int(np.shape(o)[0])
        delta=np.zeros([N_samples,self.N_stats])
        psi=np.zeros([N_samples,self.N_stats])
        path=np.zeros(N_samples)
        #initialization
        delta[0]=self.pi*self.B[0]
        psi[0]=0
        for t in range(1,N_samples):
            for s in range(self.N_stats):
                delta[t][s]=np.max(delta[t-1]*self.A[:,s])
                psi[t][s]=np.argmax(delta[t-1]*self.A[:,s])
            delta[t]=delta[t]*self.B[t]
        path[-1]=np.argmax(delta[-1])
        prob_max=np.max(delta[-1])
        #retrace and obtain the path
        for t in range(N_samples-2,-1,-1):
            path[t]=psi[t+1][int(path[t+1])]
        return prob_max,path

    '''
    ================================================================
                            #3 Learning
    gamma: 2-Dimensional (t,i) array containing P(S_t=i|O,λ)
    ksi: 3-Dimensional (t,i,j) array containing P(S_t=i,S_t+1=j|O,λ)
    ================================================================
    
    '''

    '''Baum Welch Algorithm'''
        #STEP1: Calculate the objects of iterations: gamma & ksi
        #since alpha,beta and gamma all correspond to the shape of (N_samples,N_status), we can obtain gamma by multiplying at corresponding position
    def cal_gamma(self,alpha,beta):
        gamma=alpha*beta
        gamma=gamma/np.sum(gamma,axis=1,keepdims=True) #at every t sum up all the states'
        return gamma

    def cal_ksi(self,observations,alpha,beta):
        o=observations
        N_samples=np.shape(o)[0]
        N_stats=np.shape(self.pi)[0]
        #There would be a transfer-probability container at each time point, although the data is divided by time, the probability is calculated in a consecutive manner
        ksi=np.zeros([N_samples-1,N_stats,N_stats])#3-dimensional
        for i in range(N_samples-1):
            temp=np.zeros([N_stats,N_stats])
            #extending (alpha，beta，b)'s dimensions by copying allows us to multiply by position and pass on in the form of (N_stats,N_stats) narrays
            t_alpha=np.tile(np.expand_dims(alpha[i],axis=1),(1,N_stats)) #expand_dims(axis=1):(2,1),axis=0:(1,2)#expand in columns
            t_beta=np.tile(np.expand_dims(beta[i+1],axis=0),(N_stats,1))
            t_b=np.tile(np.expand_dims(self.B[i+1],axis=0),(N_stats,1))
            temp=t_alpha*self.A*t_beta*t_b
            temp=temp/np.sum(temp) #t is the only variable right now
            ksi[i]=temp
        return ksi

    #STEP2: update pi and A
    def update_pi(self,gamma):
        return gamma[0]

    def update_A(self,gamma,ksi):
        N_stats=np.shape(gamma)[1]
        new_A=np.zeros([N_stats,N_stats])
        for i in range(N_stats):
            for j in range(N_stats):
                new_A[i][j]=np.sum(ksi[:,i,j])/np.sum(gamma[:,i])
        return new_A

    #STEP3: update B in child class



class HMM_XGB(HMM):
    def __init__(self,observations,N_stats):
        '''
       :param B: confusion matrix, it is a function here that conveys the probability of a specific state corresponding to certain observation sequence
       '''
        HMM.__init__(self,observations,N_stats) #initialize the base class first
        gen_B=np.random.random_sample((len(observations),N_stats))
        for index,b in enumerate(gen_B):
            gen_B[index]=b/sum(b)#standardization
        self.B=gen_B

    '''
    ====================================================================================
    #3 Learning (to be continued from the base class)
    #STEP3: update B in the XGBOOST setting
    ====================================================================================
    '''
    def train_XGB(self,train_data,gamma):
        '''

        -----------------------------------------------------------------------------------------------------------------------------------------------------------
        FUNCTION
        -----------------------------------------------------------------------------------------------------------------------------------------------------------
          Obtain the optimal ways to split the tree to categorize observation sequence into different hidden states (store parameters of each classifier tree)
         for further usage to apply the model to test-data to predict a resemble emission matrix (N_test_data_samples,N_stats)
        and then apply Bayesian Principles to convert the research object of the problem from
        the prob that this state accords with this observation===>>>>>> the prob that this observation is of this state(both in the shape of (N_stats,))

        INPUT
        ------
        gamma (N_samples,N_stats) ; train_data(N_samples,D)

        OUTPUT
        ------
        model that keeps a memory of the parameters about how to split a tree and how many trees are needed

        '''

        params = {
            'objective': 'multi:softprob',
            'num_class': self.N_stats}
        X=train_data

        #axis=1 the returns of labels and temp are both (N_samples,)
        label=np.array([np.argmax(i) for i in gamma])
        prob=np.array([np.max(i) for i in gamma])
        #set a threshold for the learning materials: only learn from those which have an accuracy above 90%
        thres=0.9
        while np.shape(label[prob>thres])[0]==0 and thres>0:
            #if the standard is too high to be reached, downgrade the threshold by 10%
            thres-=0.1
        label=label[prob>=thres]
        X=X[prob>=thres]
        sample_weight=prob[prob>=thres]
        xgb_train_data=xgb.DMatrix(X,label,weight=sample_weight) #unify training data X into specific matrix prepared for XGBOOST
        #training
        xgb_model=xgb.train(params,xgb_train_data,num_boost_round=20, verbose_eval=False)
        #store the current xgb model's parameters
        self.xgb=xgb_model
        return 0


    def update_XGB_B(self,observations,pi):
        """
        FUNCTION
        --------
        obtain the Emission Matrix(that has been readily converted according to Bayesian Principles)
        INPUT
        -----
        o (test_N_samples,D)
        pi (N_states,)

        OUTPUT
        ------
        B (N_samples,N_stats) P(O|S)
        *Please mind that in the seeting of XGB_HMM, the shape of Emission Matrix is dynamically different according to the length of current observation sequences
        """
        o=observations
        pred=self.xgb.predict(xgb.DMatrix(o))
        B=pred/pi #Bayesian Transfer: Likelihood/Prior Prob

        return B


    '''
    =====================================================================================
                                    #4 Training
    =====================================================================================
    '''

    #Training by One Step
    def train_step_HMM_XGB(self,train_data):
        #calculation
        alpha=self.cal_alpha(train_data)
        beta=self.cal_beta(train_data)
        ksi=self.cal_ksi(train_data,alpha,beta)
        gamma=self.cal_gamma(alpha,beta)
        #updation
        new_A=self.update_A(gamma,ksi)
        new_pi=self.update_pi(gamma)
        self.train_XGB(train_data,gamma)
        new_B=self.update_XGB_B(train_data,new_pi)
        return new_A,new_pi,new_B


    #Training until convergence or having reached the iteration limitations
    def train_HMM_XGB(self,train_data,n_iteration=20):
        new_model=HMM_XGB(train_data,self.N_stats)#Instantiate a new object
        prob_old=self.forward(train_data)
        print('The probability was', prob_old)
        for i in range(n_iteration):
            new_A,new_pi,new_B=self.train_step_HMM_XGB(train_data)
            new_model.A=new_A
            new_model.pi=new_pi
            new_model.B=new_B
            prob_new=new_model.forward(train_data)

            if prob_new<=prob_old:
                continue
            elif prob_new==float('inf'):
                break
            else:
                prob_old=prob_new
                self=new_model
                print('effective iteration:%d  prob:%f'%(i,prob_new))
        return 0


