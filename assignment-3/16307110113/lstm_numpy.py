# -*- coding: utf-8 -*-
import numpy as np
import random
import math

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(y):
    return y * (1 - y)


def tanh(x):
    return np.tanh(x)


def dtanh(y):
    return 1 - y * y


class Param:
    def __init__(self, name, value):
        self.name = name
        self.v = value #parameter value
        self.d = np.zeros_like(value) #derivative
        self.m = np.zeros_like(value) #extra information
        self.g = np.zeros_like(value)
        self.t = 0


class LSTM_numpy:
    def __init__(self, input_size, hidden_size):
        z_size = input_size + hidden_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.W_f = Param('W_f', np.random.randn(hidden_size, z_size))
        self.b_f = Param('b_f', np.zeros((hidden_size, 1)))

        self.W_i = Param('W_i', np.random.randn(hidden_size, z_size))
        self.b_i = Param('b_i', np.zeros((hidden_size, 1)))

        self.W_C = Param('W_C', np.random.randn(hidden_size, z_size))
        self.b_C = Param('b_C', np.zeros((hidden_size, 1)))

        self.W_o = Param('W_o', np.random.randn(hidden_size, z_size))
        self.b_o = Param('b_o', np.zeros((hidden_size, 1)))

        self.W_v = Param('W_v', np.random.randn(input_size, hidden_size))
        self.b_v = Param('b_v', np.zeros((input_size, 1)))
        
    def all(self):
        return [self.W_f, self.W_i, self.W_C, self.W_o, self.W_v,
               self.b_f, self.b_i, self.b_C, self.b_o, self.b_v]

    
    def forward(self, x, h_prev, C_prev):
        z = np.row_stack((h_prev, x))
        f = sigmoid(np.dot(self.W_f.v, z) + self.b_f.v)
        i = sigmoid(np.dot(self.W_i.v, z) + self.b_i.v)
        C_bar = tanh(np.dot(self.W_C.v, z) + self.b_C.v)

        C = f * C_prev + i * C_bar
        o = sigmoid(np.dot(self.W_o.v, z) + self.b_o.v)
        h = o * tanh(C)

        v = np.dot(self.W_v.v, h) + self.b_v.v
        y = np.exp(v) / np.sum(np.exp(v))

        return z, f, i, C_bar, C, o, h, v, y

    
    def backward(self, target, dh_next, dC_next, C_prev, z, f, i, C_bar, C, o, h, v, y):          
        dv = np.copy(y)
        dv[target] -= 1

        self.W_v.d += np.dot(dv, h.T)
        self.b_v.d += dv

        dh = np.dot(self.W_v.v.T, dv)        
        dh += dh_next
        do = dh * tanh(C)
        do = dsigmoid(o) * do
        self.W_o.d += np.dot(do, z.T)
        self.b_o.d += do

        dC = np.copy(dC_next)
        dC += dh * o * dtanh(tanh(C))
        dC_bar = dC * i
        dC_bar = dtanh(C_bar) * dC_bar
        self.W_C.d += np.dot(dC_bar, z.T)
        self.b_C.d += dC_bar

        di = dC * C_bar
        di = dsigmoid(i) * di
        self.W_i.d += np.dot(di, z.T)
        self.b_i.d += di

        df = dC * C_prev
        df = dsigmoid(f) * df
        self.W_f.d += np.dot(df, z.T)
        self.b_f.d += df

        dz = (np.dot(self.W_f.v.T, df)
            + np.dot(self.W_i.v.T, di)
            + np.dot(self.W_C.v.T, dC_bar)
            + np.dot(self.W_o.v.T, do))
        dh_prev = dz[:self.hidden_size, :]
        dC_prev = f * dC
        
        return dh_prev, dC_prev


    def clear_gradients(self):
        for p in self.all():
            p.d.fill(0)


    def forward_backward(self, inputs, targets, h_prev, C_prev):
        x_s, z_s, f_s, i_s,  = {}, {}, {}, {}
        C_bar_s, C_s, o_s, h_s = {}, {}, {}, {}
        v_s, y_s =  {}, {}
        
        h_s[-1] = np.copy(h_prev)
        C_s[-1] = np.copy(C_prev)
        
        loss = 0
        for t in range(len(inputs)):
            # x_s[t] = np.zeros((self.input_size, 1))
            # x_s[t][inputs[t]] = 1
            x_s[t] = inputs[t].reshape(-1, 1)
            
            (z_s[t], f_s[t], i_s[t], C_bar_s[t], C_s[t], o_s[t], h_s[t], v_s[t], y_s[t]) = self.forward(x_s[t], h_s[t - 1], C_s[t - 1])
                
            loss += -np.log(y_s[t][targets[t], 0])
            
        self.clear_gradients()

        dh_next = np.zeros_like(h_s[0])
        dC_next = np.zeros_like(C_s[0])

        for t in reversed(range(len(inputs))):
            # Backward pass
            dh_next, dC_next = self.backward(target = targets[t], dh_next = dh_next,
                                dC_next = dC_next, C_prev = C_s[t-1],
                                z = z_s[t], f = f_s[t], i = i_s[t], C_bar = C_bar_s[t],
                                C = C_s[t], o = o_s[t], h = h_s[t], v = v_s[t],
                                y = y_s[t])
            
        return loss, h_s[len(inputs) - 1], C_s[len(inputs) - 1]

    def update_sgd(self, learning_rate):
        for p in self.all():
            p.v -= learning_rate * p.d

    
    def update_Adagrad(self, learning_rate, epsilon=1e-8):
        for p in self.all():
            p.m += p.d * p.d
            p.v -= learning_rate * p.d / np.sqrt(p.m + epsilon)


    def update_RMSprop(self, learning_rate, beta, epsilon=1e-8):
        for p in self.all():
            p.m = beta * p.m + (1 - beta) * p.d * p.d
            p.v -= learning_rate * p.d / np.sqrt(p.m + epsilon)

    def update_sgd_momentum(self, learning_rate, momentum=0.9):
        for p in self.all():
            p.m = momentum * p.m - learning_rate * p.d
            p.v += p.m

    def update_Adam(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        for p in self.all():
            p.t += 1
            p.m =  beta1 * p.m + (1 - beta1) * p.d
            p.g =  beta2 * p.g + (1 - beta2) * p.d * p.d
            m_hat = p.m / (1 - np.power(beta1, p.t))
            g_hat = p.g / (1 - np.power(beta2, p.t))
            p.v -= learning_rate * m_hat / np.sqrt(g_hat + epsilon)


def gradient_check(model, inputs, targets, h_prev, C_prev):  
    _, _, _ =  model.forward_backward(inputs, targets, h_prev, C_prev)
    
    rel_err_sum = 0
    count = 0
    for param in model.all():
        #Make a copy because this will get modified
        d_copy = np.copy(param.d)

        # 检查梯度
        epsilon = 10e-8
        for i in range(param.v.shape[0]):
            for j in range(param.v.shape[1]):
                param.v[i,j] += epsilon
                loss1, _, _ =  model.forward_backward(inputs, targets, h_prev, C_prev)
                param.v[i,j] -= 2*epsilon
                loss2, _, _ =  model.forward_backward(inputs, targets, h_prev, C_prev)
                expect_grad = (loss1 - loss2) / (2 * epsilon)
                param.v[i,j] += epsilon
                print('weights(%d,%d): expected - actural %.4e - %.4e' % (
                    i, j, expect_grad, d_copy[i,j]))

                rel_err = np.abs(d_copy[i,j] - expect_grad) / np.abs(d_copy[i,j])
                rel_err_sum += rel_err
                count += 1 
                
                # If relative error is greater than 1e-06
                if rel_err > 1e-05:
                    print('%s, relative error %e > 1e-05' % (param.name, rel_err))

    print("average relative error:", rel_err_sum / count)


if __name__ == "__main__":
    test_input_size = 2
    test_hidden_size = 3
    test_sl = 4
    learning_rate = 1e-2
    numpy_model = LSTM_numpy(input_size=test_input_size, hidden_size=test_hidden_size)
    inputs = np.random.randn(test_sl, test_input_size)
    targets = np.random.randint(0, test_input_size, size=[test_sl])
    h_prev = np.zeros((test_hidden_size, 1))
    C_prev = np.zeros((test_hidden_size, 1))
    gradient_check(numpy_model, inputs, targets, h_prev, C_prev)


# Reference: http://blog.varunajayasiri.com/numpy_lstm.html

# def softmax(x):
#      return np.exp(x) / np.sum(np.exp(x)) 


# class LSTM_numpy(object):
#     def __init__(self, input_size, hidden_size, output_size):
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size

#         self.times = 0       
#         self.c_list = self.init_state_vec()
#         self.h_list = self.init_state_vec()
#         self.f_list = self.init_state_vec()
#         self.i_list = self.init_state_vec()
#         self.o_list = self.init_state_vec()
#         self.ct_list = self.init_state_vec()
#         self.net_y_list = [np.zeros((self.output_size, 1))]
#         self.y_list = [np.zeros((self.output_size, 1))]

#         self.Wfh, self.Wfx, self.bf = self.init_weight()
#         self.Wih, self.Wix, self.bi = self.init_weight()
#         self.Woh, self.Wox, self.bo = self.init_weight()
#         self.Wch, self.Wcx, self.bc = self.init_weight()
#         self.Wy = np.random.uniform(-math.sqrt(1/self.hidden_size), math.sqrt(1/self.hidden_size),
#             (self.output_size, self.hidden_size))
#         self.by = b = np.random.uniform(-math.sqrt(1/self.hidden_size), math.sqrt(1/self.hidden_size),
#             (self.output_size, 1))


#     def init_state_vec(self):
#         return [np.zeros((self.hidden_size, 1))]


#     def reset_state(self):
#         self.times = 0       
#         self.c_list = self.init_state_vec()
#         self.h_list = self.init_state_vec()
#         self.f_list = self.init_state_vec()
#         self.i_list = self.init_state_vec()
#         self.o_list = self.init_state_vec()
#         self.ct_list = self.init_state_vec()
#         self.net_y_list = [np.zeros((self.output_size, 1))]
#         self.y_list = [np.zeros((self.output_size, 1))]

#     def init_weight(self):
#         Wh = np.random.uniform(-math.sqrt(1/self.hidden_size), math.sqrt(1/self.hidden_size),
#             (self.hidden_size, self.hidden_size))
#         Wx = np.random.uniform(-math.sqrt(1/self.input_size), math.sqrt(1/self.input_size),
#             (self.hidden_size, self.input_size))
#         b = np.random.uniform(-math.sqrt(1/self.hidden_size), math.sqrt(1/self.hidden_size),
#             (self.hidden_size, 1))
#         return Wh, Wx, b

#     def linear_net(self, x, Wx, Wh, b):
#         # print(np.dot(Wh, self.h_list[self.times - 1]))
#         # print(np.dot(Wx, x))
#         # print(b)
#         return np.dot(Wh, self.h_list[self.times - 1]) + np.dot(Wx, x) + b

#     def forward(self, x):
#         self.times += 1
#         # 在门状态列表中添加当前时刻的门状态
#         fg = sigmoid(self.linear_net(x, self.Wfx, self.Wfh, self.bf))
#         self.f_list.append(fg)
#         ig = sigmoid(self.linear_net(x, self.Wix, self.Wih, self.bi))
#         self.i_list.append(ig)
#         og = sigmoid(self.linear_net(x, self.Wox, self.Woh, self.bo))
#         self.o_list.append(og)
#         ct = tanh(self.linear_net(x, self.Wcx, self.Wch, self.bc))
#         self.ct_list.append(ct)
        
#         c = fg * self.c_list[self.times - 1] + ig * ct
#         self.c_list.append(c)
#         h = og * tanh(c)
#         self.h_list.append(h)

#         net_y = np.dot(self.Wy, h) + self.by
#         self.net_y_list.append(net_y)
#         y = softmax(net_y)
#         self.y_list.append(y)

    
#     def init_weight_grad(self):
#         # 初始化权重矩阵
#         Wh_grad = np.zeros((self.hidden_size, self.hidden_size))
#         Wx_grad = np.zeros((self.hidden_size, self.input_size))
#         b_grad = np.zeros((self.hidden_size, 1))
#         return Wh_grad, Wx_grad, b_grad

#     def loss_function(self, targets):
#         if len(targets) != len(self.y_list)-1:
#             return False
#         loss = 0.0
#         for t in range(self.times, 0, -1):
#             loss += -np.log(np.exp(self.net_y_list[t][targets[t-1]])) + np.log(np.sum(np.exp(self.net_y_list[t])))
#         return loss / len(targets)


#     def backward(self, x_list, targets):
#         self.Wfh_grad, self.Wfx_grad, self.bf_grad = (self.init_weight_grad())
#         self.Wih_grad, self.Wix_grad, self.bi_grad = (self.init_weight_grad())
#         self.Woh_grad, self.Wox_grad, self.bo_grad = (self.init_weight_grad())
#         self.Wch_grad, self.Wcx_grad, self.bc_grad = (self.init_weight_grad())
#         self.Wy_grad = np.zeros((self.output_size, self.hidden_size))
#         self.by_grad = np.zeros((self.output_size, 1))

#         dh_next = np.zeros_like(self.h_list[0]) 
#         dc_next = np.zeros_like(self.c_list[0])
        
#         for t in range(self.times, 0, -1):
#             ig = self.i_list[t]
#             og = self.o_list[t]
#             fg = self.f_list[t]
#             ct = self.ct_list[t]
#             c = self.c_list[t]
#             c_prev = self.c_list[t-1]
#             h = self.h_list[t]
#             h_prev = self.h_list[t-1]
#             x = x_list[t-1]     #下标问题注意
#             net_y = self.net_y_list[t]
#             y = self.y_list[t]
#             target = targets[t-1]   #下标问题注意
            
#             dnety = np.copy(y)
#             dnety[target] -= 1

#             self.Wy_grad += np.dot(dnety, h.T)
#             self.by_grad += dnety

#             dh = np.dot(self.Wy.T, dnety)        
#             dh += dh_next
#             dog = dh * tanh(c)
#             dog = dsigmoid(og) * dog
#             self.Woh_grad += np.dot(dog, h_prev.T)
#             self.Wox_grad += np.dot(dog, x.T)
#             self.bo_grad += dog

#             dc = np.copy(dc_next)
#             dc += dh * og * dtanh(tanh(c))
#             dct = dc * ig
#             dct = dtanh(ct) * dct
#             self.Wch_grad += np.dot(dct, h_prev.T)
#             self.Wcx_grad += np.dot(dct, x.T)
#             self.bc_grad += dct

#             dig = dc * ct
#             dig = dsigmoid(ig) * dig
#             self.Wih_grad += np.dot(dig, h_prev.T)
#             self.Wix_grad += np.dot(dig, x.T)
#             self.bi_grad += dig

#             dfg = dc * c_prev
#             dfg = dsigmoid(fg) * dfg
#             self.Wfh_grad += np.dot(dfg, h_prev.T)
#             self.Wfx_grad += np.dot(dfg, x.T)
#             self.bf_grad += dfg

#             dh_prev = (np.dot(self.Wfh.T, dfg)
#                     + np.dot(self.Wih.T, dig)
#                     + np.dot(self.Wch.T, dct)
#                     + np.dot(self.Woh.T, dog))
#             dx = (np.dot(self.Wfx.T, dfg)
#                     + np.dot(self.Wix.T, dig)
#                     + np.dot(self.Wcx.T, dct)
#                     + np.dot(self.Wox.T, dog))
#             dc_prev = fg * dc

#             dh_next = dh_prev
#             dc_next = dc_prev
    

#     def update(self, learning_rate):
#         # 梯度下降
#         self.Wfh -= learning_rate * self.Wfh_grad
#         self.Wfx -= learning_rate * self.Wfx_grad
#         self.bf -= learning_rate * self.bf_grad
#         self.Wih -= learning_rate * self.Wih_grad
#         self.Wix -= learning_rate * self.Wix_grad
#         self.bi -= learning_rate * self.bi_grad
#         self.Woh -= learning_rate * self.Woh_grad
#         self.Wox -= learning_rate * self.Wox_grad
#         self.bo -= learning_rate * self.bo_grad
#         self.Wch -= learning_rate * self.Wch_grad
#         self.Wcx -= learning_rate * self.Wcx_grad
#         self.bc -= learning_rate * self.bc_grad


# def gradient_check():
#     test_input_size = 2
#     test_vocab_size = 6
#     test_hidden_size = 3
#     test_sl = 4
#     numpy_model = LSTM_numpy(test_input_size, test_hidden_size, test_vocab_size)
#     inputs = np.random.randn(test_sl, test_input_size)
#     targets = np.random.randint(0, test_vocab_size, size=[test_sl])
#     for word in inputs:
#         numpy_model.forward(word.reshape(-1, 1))
#     # 计算梯度
#     loss_numpy = numpy_model.loss_function(targets)
#     numpy_model.backward(inputs.reshape(test_sl, test_input_size, 1), targets)
#     # 检查梯度
#     epsilon = 10e-8
#     for i in range(numpy_model.Wfh.shape[0]):
#         for j in range(numpy_model.Wfh.shape[1]):
#             numpy_model.Wfh[i,j] += epsilon
#             numpy_model.reset_state()
#             for word in inputs:
#                 numpy_model.forward(word.reshape(-1, 1))
#             loss1 = numpy_model.loss_function(targets)
#             numpy_model.Wfh[i,j] -= 2*epsilon
#             numpy_model.reset_state()
#             for word in inputs:
#                 numpy_model.forward(word.reshape(-1, 1))
#             loss2 = numpy_model.loss_function(targets)
#             expect_grad = (loss1 - loss2) / (2 * epsilon)
#             numpy_model.Wfh[i,j] += epsilon
#             print('weights(%d,%d): expected - actural %.4e - %.4e' % (
#                 i, j, expect_grad, numpy_model.Wfh_grad[i,j]))
