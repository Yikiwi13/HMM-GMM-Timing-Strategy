import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import random


'''
GMM
'''
def create_GMM(mus,sigmas,ws):
    gmm=dict()
    gmm['mus']=mus
    gmm['sigmas']=sigmas
    gmm['ws']=ws
    return gmm

def Gauss_pdf(o,mu,sigma):#o is cross section data at a time point
    sigma=np.matrix(sigma)
    D=np.shape(o)[0] #the dimension of observation sequence(the num of features)
    covar_det=np.linalg.det(sigma)
    c=1/((covar_det**0.5)*(2*np.pi)**(0.5*D))
    pdf=c*np.exp(-0.5*np.dot(np.dot((o-mu),sigma.I),(o-mu).T))
    return pdf

def GMM_pdf(o,gmm):
    K,D=np.shape(gmm['mus'])
    temp=0
    for k in range(K):
        temp+=(Gauss_pdf(o,gmm['mus'][k],gmm['sigmas'][k])*gmm['ws'][k])
    return temp


    # pdfs of each GMMs
def Pdf_O2GMMs(states,o):
    '''
    :functions: to calculate the GMM-pdfs of each hidden states
    :param states: the gmm models, it is an (N,1) array containing gmm models(as dictionaries)
    :param o: observation sequence
    :return:  the probability density function of all hidden states'
    '''
    N_states = len(states)
    pdfs=np.zeros(N_states)
    for i,gmm in enumerate(states):
        pdfs[i] = GMM_pdf(o,gmm)
    return pdfs

'''
HMM_GMM Class
'''

class HMM_GMM():
    def __init__(self,observations,N_stats):
        '''

        :param observations: observation sequence
        :param self: information of the model that passes through the whole process containing components λ(π,A,B) of Hidden-Markov Model
        :param π: prior probability of state distribution(pi)
        :param A: state transition matrix
        :param B: confusion matrix, it is a function here that conveys the probability of a specific state corresponding to certain observation sequence
        :param states: contains the gmm models of all kinds of hidden states(storing the basics of gmm models like mus,sigmas)
        :param N_stats:the number of hidden states preset

        '''
        self.o=observations

        self.N_stats=N_stats

        self.A=np.random.random_sample(size=(N_stats,N_stats))
        self.A=self.A/np.sum(self.A,axis=1)

        self.pi=np.random.random_sample(N_stats)
        self.pi=self.pi/float(np.sum(self.pi))

        #initialization of confusion matrix

        states=[]
        D=np.shape(self.o)[1] #dimension of observation sequence
        for i in range(N_stats):
            mus=0.9*np.random.random_sample([N_stats,D]) #3 Gaussian components in one gmm; N维特征值
            sigmas=np.array([np.eye(D,D) for v in range(N_stats)]) #hypothesize the features are interindependent
            ws=np.random.random_sample(N_stats)
            ws=ws/float(np.sum(ws))
            gmm=create_GMM(mus,sigmas,ws)
            states.append(gmm)
        self.states=states
        self.B=Pdf_O2GMMs

    '''
    Evaluation: evaluating the degree of matching between the observation sequence and the HMM model trying to depict it:P(O|λ,S)
    The observation sequence should be renewed data each time evaluating, so the observation is not the preset observation sequence 
    '''
    '''(I) Forward Algorithm'''
    def cal_alpha(self,observations):
        o=observations
        N_samples=np.shape(o)[0]
        N_stats=np.shape(self.pi)[0]
        #initialization
        alpha=np.zeros([N_samples,N_stats])
        alpha[0]=self.pi*self.B(self.states,o[0])
        #iteration
        for t in range(1,N_samples):
            temp=np.dot(alpha[t-1],self.A)
            alpha[t]=temp*self.B(self.states,o[t])
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
            beta[t]=np.dot(beta[t+1],self.A.T)*self.B(self.states,o[t+1])
        return beta

    def backward(self,observations):
        o=observations
        beta=self.cal_beta(o)
        prob_seq_b=np.dot(beta[0]*self.B(self.states,o[0]),self.pi)
        prob_seq_b=np.log(prob_seq_b)
        return prob_seq_b

    '''
    Decoding
    '''
    def Viterbi(self,test):
        o=test
        N_samples=np.shape(o)[0]
        N_stats=np.shape(self.pi)[0]

        delta=np.zeros([N_samples,N_stats]) #the maximum probability up to this state at this time point
        psi=np.zeros([N_samples,N_stats]) #the most likely last hidden state this point is past on from
        path=np.zeros(N_samples)#retrace the path

        #initialization
        delta[0]=self.pi*self.B(self.states,o[0])
        psi[0]=0
        for t in range(1,N_samples):
            for s in range(N_stats):
                delta[t][s]=np.max(delta[t-1]*self.A[:,s])
                psi[t][s]=np.argmax(delta[t-1]*self.A[:,s])
            delta[t]=delta[t]*self.B(self.states,o[t])
        path[-1]=np.argmax(delta[-1])
        prob_max=np.max(delta[-1])
        #obatin the path by retracing along the psi
        for t in range(N_samples-2,-1,-1):
            path[t]=psi[t+1][int(path[t+1])]
        return prob_max,path


    '''
    Learning
    ================================================================
    Baum Welch Algorithm
    gamma: 2-Dimensional (t,i) array containing P(S_t=i|O,λ)
    ksi: 3-Dimensional (t,i,j) array containing P(S_t=i,S_t+1=j|O,λ)
    ================================================================
    '''

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
            t_b=np.tile(np.expand_dims(self.B(self.states,o[i+1]),axis=0),(N_stats,1))
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

    #STEP3: update B (with GMM models)
    '''
    The param in GMM that needs to be updated:gamma_mix(3-D),P(St=i and belong to m component in GMMi|O,λ)
    '''
    def update_GMM_B(self,train_data,gamma):
        T,D=np.shape(train_data)
        N_mix=self.N_stats #number of components set in each GMM model(here the default is the same with the number of hidden states)
        N_stats=self.N_stats
        gamma_mix=np.zeros([T,N_stats,N_mix])
        for t in range(T):
            for s in range(N_stats):
                p_mix=np.zeros(N_mix)
                for m in range(N_mix):
                    p_mix[m]=Gauss_pdf(train_data[t],self.states[s]['mus'][m],self.states[s]['sigmas'][m])
                    p_mix[m]=p_mix[m]*self.states[s]['ws'][m]
                p_mix=p_mix/np.sum(p_mix)
                gamma_mix[t,s,:]=p_mix*gamma[t][s]

        new_states=[]
        for s in range(N_stats): #each hidden state has a GMM model
            gmm=dict()
            gmm['ws']=np.zeros(N_mix)
            gmm['mus']=np.zeros([N_mix,D])
            gmm['sigmas']=np.zeros([N_mix,D,D])
            new_states.append(gmm)
        for s in range(N_stats):
            for m in range(N_mix):
                r_k=gamma_mix[:,s,m]
                N_k=np.sum(r_k)
                r_k=r_k[:,np.newaxis] #transfer into (n,1) array
                #update mu
                mu=np.sum(train_data*r_k,axis=0)/N_k
                #update sigma
                dx=train_data-self.states[s]['mus'][m]#all-in calculation
                sigma=np.zeros([D,D])
                for t in range(T):
                    sigma=sigma+r_k[t,0]*np.outer(dx[t],dx[t])
                sigma=sigma/N_k
                sigma=sigma+np.eye(D)*0.001# avoid singular matrix
                w=N_k/T

                new_states[s]['mus'][m]=mu
                new_states[s]['sigmas'][m]=sigma
                new_states[s]['ws'][m]=w
            new_states[s]['ws']=new_states[s]['ws']/np.sum(new_states[s]['ws'])
        return new_states

    '''
    
    Training
    
    '''
    #train by one step
    def train_step_GMM_HMM(self,train_data):
        alpha=self.cal_alpha(train_data)
        beta=self.cal_beta(train_data)
        ksi=self.cal_ksi(train_data,alpha,beta)
        gamma=self.cal_gamma(alpha,beta)
        new_A=self.update_A(gamma,ksi)
        new_pi=self.update_pi(gamma)
        new_states=self.update_GMM_B(train_data,gamma)
        return new_A,new_pi,new_states


    #train in general until convergence or reaches the ceiling of iterations
    def train_GMM_HMM(self,train_data,n_iteration):
        new_model=HMM_GMM(train_data,self.N_stats)
        prob_old=self.forward(train_data)
        print('The probability was', prob_old)
        for i in range(n_iteration):
            new_A,new_pi,new_states=self.train_step_GMM_HMM(train_data)
            new_model.A=new_A
            new_model.pi=new_pi
            new_model.states=new_states
            new_model.B=Pdf_O2GMMs
            prob_new=new_model.forward(train_data)
            print('iteration:%d  prob:%f'%(i,prob_new))
            if prob_new>prob_old:
                if prob_new-prob_old<0.5:
                    break
                else:
                    prob_old=prob_new
                    self=new_model
            else:
                break #convergence
        return 0
