import numpy as np
import sys 
import os
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('PRML\\assignment3\\src\\')
from Configure import Config
import torch.nn.functional as F

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def dtanh(y):
    return 1 - y * y

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

class numpy_lstm():
    Wf = None
    Wi = None
    Wc = None
    Wo = None
    bf = None
    bi = None
    bc = None
    bc = None
    bo = None
    input_dim = 128
    hidden_dim = 256
    feature_size = input_dim+hidden_dim
    def __init__(self,conf:Config):
        self.learning_rate = conf.learning_rate
        # z=[x,hx] |z|=feature_size
        self.input_dim = conf.embedding_dim
        self.hidden_dim = conf.hidden_dim
        self.feature_size = self.input_dim+self.hidden_dim
        self.Wf=np.random.randn(self.feature_size,self.hidden_dim,) / np.sqrt(self.feature_size / 2.)
        self.Wi=np.random.randn(self.feature_size,self.hidden_dim) / np.sqrt(self.feature_size / 2.)
        self.Wc=np.random.randn(self.feature_size,self.hidden_dim) / np.sqrt(self.feature_size / 2.)
        self.Wo=np.random.randn(self.feature_size,self.hidden_dim) / np.sqrt(self.feature_size / 2.)
        self.bf=np.zeros((1,self.hidden_dim))
        self.bi=np.zeros((1,self.hidden_dim))
        self.bc=np.zeros((1,self.hidden_dim))
        self.bo=np.zeros((1,self.hidden_dim))

        self.Wy=np.random.randn(self.hidden_dim,self.input_dim) / np.sqrt(self.input_dim / 2.)
        self.by=np.zeros((1,self.input_dim))

        self.unit_derivatives = []
        self.Loss = None

    def unit_forward(self,input,hidden):
        # input: (batch_size * embedding_size)
        # hidden=(hx,cx): batch_size*hidden_size *2
        hx,cx = hidden
        input_z = np.concatenate([input,hx],axis=1)
        # input = np.column_stack((input,hx))
        f_gate = input_z @ self.Wf + self.bf # bs X fs · fs X hs + 1 X hs= bs X hs
        f_gate = sigmoid(f_gate)
        i_gate = input_z @ self.Wi + self.bi
        i_gate = sigmoid(i_gate)
        o_gate = input_z @ self.Wo + self.bo
        o_gate = sigmoid(o_gate)
        c_gate = input_z @ self.Wc + self.bc
        c_gate = tanh(c_gate)

        cx = f_gate * cx + i_gate * c_gate
        hx = o_gate * tanh(cx)
        self.unit_derivation(input,hidden,f_gate,i_gate,o_gate,c_gate,cx,hx)

        return hx,cx
    
    def forward(self,input,hidden=None):
        print(input.shape)
        # input: seq_len * batch_size * embedding_size
        seq_len = input.shape[0]
        batch_size = input.shape[1]
        if hidden is None:
            zeros = np.zeros((batch_size,self.hidden_dim))
            hx,cx = (zeros,zeros)
            hidden = (hx,cx)
        output = []
        self.unit_derivatives = []
        for i in range(seq_len):
            hx,cx = self.unit_forward(input[i,:,:],hidden)
            hidden = hx,cx
            output.append(hx)
        # np.concatenate(output): (seq_len , batch_size , hidden_size)
        self.H_T = np.concatenate(output)
        return np.concatenate(output) @ self.Wy + self.by #(seq*len*batch_size,embedding_size)
    
    def unit_derivation(self,input,hidden,f_t,i_t,o_t,cg_t,c_t,h_t):
        h_t_1,c_t_1 = hidden
        z = np.concatenate([input,h_t_1],axis=1) # (batch_size,hidden_size)

        unit_d = {} #frac{\partial h_t }{\partial * }
        unit_d['dc_t'] = o_t*(1-tanh(c_t)**2)
        unit_d['do_t'] = tanh(c_t)
        unit_d['dcg_t'] = unit_d['dc_t']*i_t
        unit_d['dc_t_1'] = unit_d['dc_t']*f_t
        unit_d['di_t'] = unit_d['dc_t']*cg_t
        unit_d['df_t'] = unit_d['dc_t']*c_t_1

        tmpdo = (unit_d['do_t']*o_t*(1-o_t)) @ self.Wo.T
        tmpdf = ( o_t*(1-tanh(c_t)**2 )*c_t_1*f_t*(1-f_t) ) @ self.Wf.T
        tmpdi = ( o_t*(1-tanh(c_t)**2 )*cg_t*i_t*(1-i_t) ) @ self.Wi.T
        tmpdc = ( o_t*(1-tanh(c_t)**2 )*i_t*(1-cg_t**2) ) @ self.Wc.T
        unit_d['dz'] =  tmpdo + tmpdf + tmpdi + tmpdc
        unit_d['dh'] = unit_d['dz'][:,:input.shape[1]] #0:embedding_size

        unit_d['dbf'] = unit_d['df_t'] * f_t * (1 - f_t)
        unit_d['dbi'] = unit_d['di_t'] * i_t * (1 - i_t)
        unit_d['dbc'] = unit_d['dcg_t'] * (1 - cg_t**2)
        unit_d['dbo'] = unit_d['do_t'] * o_t * (1-o_t)

        unit_d['dwf'] = z.T @ unit_d['dbf']
        unit_d['dwi'] = z.T @ unit_d['dbi']
        unit_d['dwc'] = z.T @ unit_d['dbc']
        unit_d['dwo'] = z.T @ unit_d['dbo']
        self.unit_derivatives.append(unit_d)

    def partial_L_h_t(self,partial_L_T,cache):
        return partial_L_T * cache

    def backword(self, y_out, y_true):
    	y_pred = softmax(y_pred)
        #self.Loss = y_true.T * np.log(y_pred)
        target = y_true.reshape(-1,1)
        
        # self.Loss = 0
        # for i in range(y_pred.shape[0]) :
        #     self.Loss += y_pred[target[i][0]]

        self.Loss = -np.sum( self.target * np.log(y_pred) )
        
        tmp = y_pred - y_true
        partial_L_T = ( y_pred - y_true ) @ self.Wy.T
        # p = self.softmax( W.dot(X.T) + b )
        # tmp = p - y
        # wgradient = (tmp.dot(X) )/X.shape[0]
        # bgradient = (np.sum(tmp,axis = 1,keepdims = True) / X.shape[0] )

        # partial_L_T = ( target * (1-y_pred) ) @ self.Wy.T
        partial_L_b = {'dWc':[],'dWf':[],'dWo':[],'dWi':[]}
        partial_L_W = {'dbc':[],'dbf':[],'dbo':[],'dbi':[]}
        seq_len = self.Loss.shape[0]
        cached = partial_L_T
        for i in range(0,seq_len):
            print(i)
            partial_L_b['dWc'].append(cached*self.unit_derivatives[seq_len-1-i]['dwc']) 
            partial_L_b['dWf'].append(cached*self.unit_derivatives[seq_len-1-i]['dwf']) 
            partial_L_b['dWi'].append(cached*self.unit_derivatives[seq_len-1-i]['dwi']) 
            partial_L_b['dWo'].append(cached*self.unit_derivatives[seq_len-1-i]['dwo']) 

            partial_L_W['dbc'].append(cached*self.unit_derivatives[seq_len-1-i]['dbc']) 
            partial_L_W['dbf'].append(cached*self.unit_derivatives[seq_len-1-i]['dbf']) 
            partial_L_W['dbi'].append(cached*self.unit_derivatives[seq_len-1-i]['dbi']) 
            partial_L_W['dbo'].append(cached*self.unit_derivatives[seq_len-1-i]['dbo']) 

            cached = partial_L_T * cached*self.unit_derivatives[seq_len-1-i]['dh']
        
        deta_Wf = np.sum(partial_L_W['dWf'])
        deta_bf = np.sum(partial_L_b['dbf'])
        deta_Wi = np.sum(partial_L_W['dWi'])
        deta_bi = np.sum(partial_L_b['dbi'])
        deta_Wo = np.sum(partial_L_W['dWo'])
        deta_bo = np.sum(partial_L_b['dbo'])
        deta_Wc = np.sum(partial_L_W['dWc'])
        deta_bc = np.sum(partial_L_b['dbc'])

        # deta_by = y_pred*(1-y_pred)
        deta_by = y_pred - y_true
        deta_Wy = self.H_T.T @ deta_by

        self.Wc -= self.learning_rate * deta_Wc
        self.Wf -= self.learning_rate * deta_Wf
        self.Wi -= self.learning_rate * deta_Wi
        self.Wo -= self.learning_rate * deta_Wo

        self.bc -= self.learning_rate * deta_bc
        self.bf -= self.learning_rate * deta_bf
        self.bi -= self.learning_rate * deta_bi
        self.bo -= self.learning_rate * deta_bo

        self.Wy -= self.learning_rate * deta_Wy
        self.by -= self.learning_rate * deta_by













if __name__ == "__main__":
    conf = Config()
    lstm = numpy_lstm(conf)
    input = np.random.randn(5,32,128)
    target = np.random.randint(0,128,(5,32))
    out = lstm.forward(input)
    prob = softmax(out)
    print(out.shape)
    print(out)
    print(prob.shape)
    lstm.backword(prob,target)










