import numpy as np

class Sigmoid(object):
  def forward(self, weighted_input):
    return 1.0 / (1.0+np.exp(-weighted_input))

  def backward(self, output):
    return output * (1-output)

class Tanh(object):
  def forward(self, weighted_input):
    return 2.0 / (1.0+np.exp(-2*weighted_input)) - 1.0
  
  def backward(self, output):
    return 1 - output*output

class LstmLayer(object):
  def __init__(self, feat_s, hidden_dim, learning_rate):
    self.feat_s = feat_s
    self.hidden_dim = hidden_dim
    self.learning_rate = learning_rate

    self.gate_activator = Sigmoid()
    self.output_activator = Tanh()

    self.times = 0

    # 保存各个时刻的状态
    self.f_list = self.init_state_vec()
    self.i_list = self.init_state_vec()
    self.ch_list = self.init_state_vec()
    self.o_list = self.init_state_vec()
    self.c_list = self.init_state_vec()
    self.h_list = self.init_state_vec()

    # 权重初始化
    self.Wfh, self.Wfx, self.bf = self.init_weight_mat()
    self.Wih, self.Wix, self.bi = self.init_weight_mat()
    self.Woh, self.Wox, self.bo = self.init_weight_mat()
    self.Wch, self.Wcx, self.bc = self.init_weight_mat()

  def init_state_vec(self):
    state_vec_list = []
    state_vec_list.append(np.zeros((self.hidden_dim, 1)))
    return state_vec_list
  
  def init_weight_mat(self):
    Wh = np.random.uniform(-1e-4, 1e-4, (self.hidden_dim, self.hidden_dim))
    Wx = np.random.uniform(-1e-4, 1e-4, (self.hidden_dim, self.feat_s))
    b = np.zeros((self.hidden_dim, 1))
    return Wh, Wx, b

  def forward(self, x):
    self.times += 1

    ft = self.calc_gate(x, self.Wfh, self.Wfx, self.bf, self.gate_activator)
    self.f_list.append(ft)

    it = self.calc_gate(x, self.Wih, self.Wix, self.bi, self.gate_activator)
    self.i_list.append(it)

    cht = self.calc_gate(x, self.Wch, self.Wcx, self.bc, self.output_activator)
    self.ch_list.append(cht)

    ot = self.calc_gate(x, self.Woh, self.Wox, self.bo, self.gate_activator)
    self.o_list.append(ot)

    ct = ft * self.c_list[self.times-1] + it * cht
    self.c_list.append(ct)

    ht = ot * self.output_activator.forward(ct)
    self.h_list.append(ht)

  def calc_gate(self, x, Wh, Wx, b, activator):
    h = self.h_list[self.times-1]
    return activator.forward(np.dot(Wh, h) + np.dot(Wx, x) + b)

  def backward(self, x, delta_h):
    self.calc_delta(delta_h)
    self.calc_gradient(x)

  def calc_delta(self, delta_h):
    self.delta_f_list = self.init_delta()
    self.delta_i_list = self.init_delta()
    self.delta_ch_list = self.init_delta()
    self.delta_o_list = self.init_delta()

    self.delta_h_list = self.init_delta()
    self.delta_h_list[-1] = delta_h

    for k in range(self.times, 0, -1):
      self.calc_delta_k(k)

  def init_delta(self):
    delta_list = []
    for i in range(self.times+1):
      delta_list.append(np.zeros((self.hidden_dim, 1)))
    return delta_list

  def calc_delta_k(self, k):
    ft = self.f_list[k]
    it = self.i_list[k]
    cht = self.ch_list[k]
    ot = self.o_list[k]
    ct = self.c_list[k]
    c_prev = self.c_list[k-1]
    tanh_ct = self.output_activator.forward(ct)
    delta_k = self.delta_h_list[k]

    delta_ft = delta_k * ot * (1-tanh_ct*tanh_ct) * c_prev * self.gate_activator.backward(ft)
    delta_it = delta_k * ot * (1-tanh_ct*tanh_ct) * ct * self.gate_activator.backward(it)
    delta_cht = delta_k * ot * (1-tanh_ct*tanh_ct) * it * self.output_activator.backward(cht)
    delta_ot = delta_k * tanh_ct * self.gate_activator.backward(ot)
    delta_h_prev = (
      np.dot(delta_ft.transpose(), self.Wfh) +
      np.dot(delta_it.transpose(), self.Woh) +
      np.dot(delta_cht.transpose(), self.Wch) + 
      np.dot(delta_ot.transpose(), self.Woh)
    ).transpose()

    self.delta_f_list[k] = delta_ft
    self.delta_i_list[k] = delta_it
    self.delta_ch_list[k] = delta_cht
    self.delta_o_list[k] = delta_ot
    self.delta_h_list[k-1] = delta_h_prev

  def calc_gradient(self, x):
    self.Wfh_grad, self.Wfx_grad, self.bf_grad = self.init_weight_grad_mat()
    self.Wih_grad, self.Wix_grad, self.bi_grad = self.init_weight_grad_mat()
    self.Wch_grad, self.Wcx_grad, self.bc_grad = self.init_weight_grad_mat()
    self.Woh_grad, self.Wox_grad, self.bo_grad = self.init_weight_grad_mat()

    for t in range(self.times, 0, -1):
      Wfh_grad, bf_grad, Wih_grad, bi_grad, Wch_grad, bc_grad, Woh_grad, bo_grad = self.calc_grad_t(t)

      self.Wfh_grad += Wfh_grad
      self.bf_grad += bf_grad

      self.Wih_grad += Wih_grad
      self.bi_grad += bi_grad

      self.Wch_grad += Wch_grad
      self.bc_grad += bc_grad

      self.Woh_grad += Woh_grad
      self.bo_grad += bo_grad

      print("---------------%d-------------" % t)
      print(Wfh_grad)
      print(self.Wfh_grad)

    xt = x.transpose()
    self.Wfh_grad = np.dot(self.delta_f_list[-1], xt)
    self.Wih_grad = np.dot(self.delta_i_list[-1], xt)
    self.Wch_grad = np.dot(self.delta_ch_list[-1], xt)
    self.Woh_grad = np.dot(self.delta_o_list[-1], xt)   

  def init_weight_grad_mat(self):
    Wh_grad = np.zeros((self.hidden_dim, self.hidden_dim))
    Wx_grad = np.zeros((self.hidden_dim, self.feat_s))
    b_grad = np.zeros((self.hidden_dim, 1))
    return Wh_grad, Wx_grad, b_grad

  def calc_grad_t(self, t):
    h_prev = self.h_list[t-1].transpose()

    Wfh_grad = np.dot(self.delta_f_list[t], h_prev)
    bf_grad = self.delta_f_list[t]

    Wih_grad = np.dot(self.delta_i_list[t], h_prev)
    bi_grad = self.delta_i_list[t]

    Wch_grad = np.dot(self.delta_ch_list[t], h_prev)
    bc_grad = self.delta_ch_list[t]

    Woh_grad = np.dot(self.delta_o_list[t], h_prev)
    bo_grad = self.delta_o_list[t]
    return Wfh_grad, bf_grad, Wih_grad, bi_grad, Wch_grad, bc_grad, Woh_grad, bo_grad

  def update(self):
    self.Wfh -= self.learning_rate * self.Wfh_grad
    self.Wfx -= self.learning_rate * self.Wfx_grad
    self.bf -= self.learning_rate * self.bf_grad

    self.Wih -= self.learning_rate * self.Wih_grad
    self.Wix -= self.learning_rate * self.Wix_grad
    self.bi -= self.learning_rate * self.bi_grad

    self.Wch -= self.learning_rate * self.Wch_grad
    self.Wcx -= self.learning_rate * self.Wcx_grad
    self.bc -= self.learning_rate * self.bc_grad

    self.Woh -= self.learning_rate * self.Woh_grad
    self.Wox -= self.learning_rate * self.Wox_grad
    self.bo -= self.learning_rate * self.bo_grad    

  def reset_state(self):
    self.times = 0

    self.f_list = self.init_state_vec()
    self.i_list = self.init_state_vec()
    self.ch_list = self.init_state_vec()
    self.o_list = self.init_state_vec()
    self.c_list = self.init_state_vec()
    self.h_list = self.init_state_vec()

      
def data_set():
  x = [np.array([[1], [2], [3]]),
       np.array([[2], [3], [4]])]
  d = np.array([[1], [2]])
  return x, d

def gradient_check():
    '''
    梯度检查
    '''
    # 设计一个误差函数，取所有节点输出项之和
    error_function = lambda o: o.sum()
    lstm = LstmLayer(3, 2, 1e-3)
    # 计算forward值
    x, d = data_set()
    lstm.forward(x[0])
    lstm.forward(x[1])
    # 求取sensitivity map
    sensitivity_array = np.ones(lstm.h_list[-1].shape,
                                dtype=np.float64)
    # 计算梯度
    lstm.backward(x[1], sensitivity_array)
    # 检查梯度
    epsilon = 10e-4
    for i in range(lstm.Wfh.shape[0]):
        for j in range(lstm.Wfh.shape[1]):
            lstm.Wfh[i,j] += epsilon
            lstm.reset_state()
            lstm.forward(x[0])
            lstm.forward(x[1])
            err1 = error_function(lstm.h_list[-1])
            lstm.Wfh[i,j] -= 2*epsilon
            lstm.reset_state()
            lstm.forward(x[0])
            lstm.forward(x[1])
            err2 = error_function(lstm.h_list[-1])
            expect_grad = (err1 - err2) / (2 * epsilon)
            lstm.Wfh[i,j] += epsilon
            print('weights(%d,%d): expected - actural %.4e - %.4e' % (
                i, j, expect_grad, lstm.Wfh_grad[i,j]))
    return lstm

gradient_check()