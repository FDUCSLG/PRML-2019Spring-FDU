import numpy as np
import math
import torch
import torch.nn as nn

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_back(x):
    return x*(1-x)
    
def tanh_back(x):
    return 1-x*x

def sqr(x):
    return x*x

class LSTMLayer:
    def __init__(self,input_size,hidden_size):
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.times=0
        self.inputs=None
        self.c_t=self.init_state()
        self.h_t=self.init_state()
        self.f_t=self.init_state()
        self.i_t=self.init_state()
        self.o_t=self.init_state()
        self.ct_t=self.init_state()
        
        self.Wfh,self.Wfx,self.bf=self.init_weight_mat()
        self.Wih,self.Wix,self.bi=self.init_weight_mat()
        self.Woh,self.Wox,self.bo=self.init_weight_mat()
        self.Wch,self.Wcx,self.bc=self.init_weight_mat()
        
        self.Wfh_grad,self.Wfx_grad,self.bf_grad=self.init_weight_gradient()
        self.Wih_grad,self.Wix_grad,self.bi_grad=self.init_weight_gradient()
        self.Woh_grad,self.Wox_grad,self.bo_grad=self.init_weight_gradient()
        self.Wch_grad,self.Wcx_grad,self.bc_grad=self.init_weight_gradient()
        
    def init_state(self):
        state_vecs=[np.zeros((self.hidden_size,1))]
        return state_vecs
        
    def init_weight_mat(self):
        Wh=np.random.uniform(-1/math.sqrt(self.hidden_size),1/math.sqrt(self.hidden_size),(self.hidden_size,self.hidden_size))
        Wx=np.random.uniform(-1/math.sqrt(self.hidden_size),1/math.sqrt(self.hidden_size),(self.hidden_size,self.input_size))
        b=np.zeros((self.hidden_size,1))
        return Wh,Wx,b
        
    def forward(self,inputs,state=None):
        self.inputs=inputs
        length,_ = inputs.shape
        self.times=0
        self.c_t=self.init_state()
        self.h_t=self.init_state()
        self.f_t=self.init_state()
        self.i_t=self.init_state()
        self.o_t=self.init_state()
        self.ct_t=self.init_state()
        if state is not None:
            self.h_t[0],self.c_t[0]=state
        
        for i in range(length):
            self.times+=1
            x=inputs[i,:,np.newaxis]
            ft=self.calc_gate(x,self.Wfx,self.Wfh,self.bf,sigmoid)
            self.f_t.append(ft)
            it=self.calc_gate(x,self.Wix,self.Wih,self.bi,sigmoid)
            self.i_t.append(it)
            ot=self.calc_gate(x,self.Wox,self.Woh,self.bo,sigmoid)
            self.o_t.append(ot)
            ctt=self.calc_gate(x,self.Wcx,self.Wch,self.bc,np.tanh)
            self.ct_t.append(ctt)
            c=ft*self.c_t[self.times-1]+it*ctt
            self.c_t.append(c)
            h=ot*np.tanh(c)
            self.h_t.append(h)
        return self.h_t,(h,c)
        
    def calc_gate(self,x,Wx,Wh,b,activator):
        h=self.h_t[self.times-1]
        net=np.dot(Wh,h)+np.dot(Wx,x)+b
        gate=activator(net)
        return gate
    
    def backward(self,t,delta):
        self.calc_delta(delta,t)
        self.calc_gradient(self.inputs[t],t)
        
    def calc_delta(self,delta_h,t):
        self.delta_h_t=self.init_delta()  
        self.delta_o_t=self.init_delta()
        self.delta_i_t=self.init_delta()
        self.delta_f_t=self.init_delta()
        self.delta_ct_t=self.init_delta()
        self.delta_h_t[t+1]=delta_h
        for k in range(t+1,0,-1):
            self.calc_delta_k(k)
            
    def init_delta(self):
        delta_t=[]
        for i in range(self.times+1):
            delta_t.append(np.zeros((self.hidden_size,1)))
        return delta_t
        
    def calc_delta_k(self,k):
        ig=self.i_t[k]
        og=self.o_t[k]
        fg=self.f_t[k]
        ct=self.ct_t[k]
        c=self.c_t[k]
        c_prev=self.c_t[k-1]
        cc=np.tanh(c)
        delta_k=self.delta_h_t[k]
        delta_o=(delta_k*cc*sigmoid_back(og))
        delta_f=(delta_k*og*(1-sqr(cc))*c_prev*sigmoid_back(fg))
        delta_i=(delta_k*og*(1-sqr(cc))*ct*sigmoid_back(ig))
        delta_ct=(delta_k*og*(1-sqr(cc))*ig*tanh_back(ct))
        delta_h_prev=(
                np.dot(delta_o.transpose(),self.Woh)+
                np.dot(delta_i.transpose(),self.Wih)+
                np.dot(delta_f.transpose(),self.Wfh)+
                np.dot(delta_ct.transpose(),self.Wch)
            ).transpose()

        self.delta_h_t[k-1]=delta_h_prev
        self.delta_f_t[k]=delta_f
        self.delta_i_t[k]=delta_i
        self.delta_o_t[k]=delta_o
        self.delta_ct_t[k]=delta_ct
        
    def clear_gradient(self):
        self.Wfh_grad,self.Wfx_grad,self.bf_grad=self.init_weight_gradient()
        self.Wih_grad,self.Wix_grad,self.bi_grad=self.init_weight_gradient()
        self.Woh_grad,self.Wox_grad,self.bo_grad=self.init_weight_gradient()
        self.Wch_grad,self.Wcx_grad,self.bc_grad=self.init_weight_gradient()
        
    def calc_gradient(self,x,t):
        for t in range(t+1,0,-1):
            Wfh_grad,bf_grad, Wih_grad,bi_grad, Woh_grad,bo_grad, Wch_grad,bc_grad=self.calc_gradient_t(t)
            self.Wfh_grad+=Wfh_grad
            self.bf_grad+=bf_grad
            self.Wih_grad+=Wih_grad
            self.bi_grad+=bi_grad
            self.Woh_grad+=Woh_grad
            self.bo_grad+=bo_grad
            self.Wch_grad+=Wch_grad
            self.bc_grad+=bc_grad

        xt=x[:,np.newaxis].transpose()
        self.Wfx_grad+=np.dot(self.delta_f_t[-1],xt)
        self.Wix_grad+=np.dot(self.delta_i_t[-1],xt)
        self.Wox_grad+=np.dot(self.delta_o_t[-1],xt)
        self.Wcx_grad+=np.dot(self.delta_ct_t[-1],xt)
        
    def init_weight_gradient(self):
        Wh_grad=np.zeros((self.hidden_size,self.hidden_size))
        Wx_grad=np.zeros((self.hidden_size,self.input_size))
        b_grad=np.zeros((self.hidden_size,1))
        return Wh_grad,Wx_grad,b_grad
        
    def calc_gradient_t(self,t):
        h_prev=self.h_t[t-1].transpose()
        Wfh_grad=np.dot(self.delta_f_t[t],h_prev)
        bf_grad=self.delta_f_t[t]
        Wih_grad=np.dot(self.delta_i_t[t],h_prev)
        bi_grad=self.delta_f_t[t]
        Woh_grad=np.dot(self.delta_o_t[t],h_prev)
        bo_grad=self.delta_f_t[t]
        Wch_grad=np.dot(self.delta_ct_t[t],h_prev)
        bc_grad=self.delta_ct_t[t]
        return Wfh_grad,bf_grad,Wih_grad,bi_grad, Woh_grad,bo_grad,Wch_grad,bc_grad
        
    def reset_state(self):
        self.times=0
        self.inputs=None
        self.c_t=self.init_state()
        self.h_t=self.init_state()
        self.f_t=self.init_state()
        self.i_t=self.init_state()
        self.o_t=self.init_state()
        self.ct_t=self.init_state()
        
lstm=LSTMLayer(3,2)
x=np.array([[0.1,-0.2,0.3],[-0.1,0.2,0.3]])
lstm.forward(x)
delta = np.ones(lstm.h_t[-1].shape, dtype=np.float64) # suppose function is the sum of the last h
print(lstm.times)
lstm.backward(1,delta)
epsilon = 1e-4

print("check gradient of Wfx")
for i in range(lstm.Wfx.shape[0]):
    for j in range(lstm.Wfx.shape[1]):
        lstm.Wfx[i,j]+=epsilon
        lstm.reset_state()
        lstm.forward(x)
        err1 = np.sum(lstm.h_t[-1])
        
        lstm.Wfx[i,j]-=2*epsilon
        lstm.reset_state()
        lstm.forward(x)
        err2=np.sum(lstm.h_t[-1])
        expect_grad=(err1-err2) /(2*epsilon)
        lstm.Wfx[i,j]+=epsilon
        print('weights(%d,%d): expected, actual: %.4e, %.4e' % (i, j, expect_grad, lstm.Wfx_grad[i,j]))

print()                
print("check gradient of Wfh")
for i in range(lstm.Wfh.shape[0]):
    for j in range(lstm.Wfh.shape[1]):
        lstm.Wfh[i,j]+=epsilon
        lstm.reset_state()
        lstm.forward(x)
        err1 = np.sum(lstm.h_t[-1])
        
        lstm.Wfh[i,j]-=2*epsilon
        lstm.reset_state()
        lstm.forward(x)
        err2=np.sum(lstm.h_t[-1])
        expect_grad=(err1-err2)/(2*epsilon)
        lstm.Wfh[i,j]+=epsilon
        print('weights(%d,%d): expected, actual: %.4e, %.4e' % (i, j, expect_grad, lstm.Wfh_grad[i,j]))                    
