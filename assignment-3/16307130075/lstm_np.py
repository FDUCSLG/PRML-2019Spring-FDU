import numpy as np


def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def tanh(x):
	return np.tanh(x)

class Parameter:
	def __init__(self, val):
		self.v = val
		self.g = np.zeros_like(val)
		self.m = np.zeros_like(val)

class LSTM:
	def __init__(self, x_size, h_size, o_size):
		self.x_size = x_size
		self.h_size = h_size
		self.o_size = o_size
		self.wf = Parameter(np.random.rand(h_size, x_size + h_size))
		self.bf = Parameter(np.zeros([h_size, 1]))
		self.wi = Parameter(np.random.rand(h_size, x_size + h_size))
		self.bi = Parameter(np.zeros([h_size, 1]))
		self.wc = Parameter(np.random.rand(h_size, x_size + h_size))
		self.bc = Parameter(np.zeros([h_size, 1]))
		self.wo = Parameter(np.random.rand(h_size, x_size + h_size))
		self.bo = Parameter(np.zeros([h_size, 1]))
		self.w = Parameter(np.random.rand(o_size, h_size))
		self.b = Parameter(np.zeros([o_size, 1]))

	def zero_grad(self):
		self.wf.g.fill(0)
		self.bf.g.fill(0)
		self.wi.g.fill(0)
		self.bi.g.fill(0)
		self.wc.g.fill(0)
		self.bc.g.fill(0)
		self.wo.g.fill(0)
		self.bo.g.fill(0)
		self.w.g.fill(0)
		self.b.g.fill(0)

	def sgd_update(self, param, lr):
		param.v = param.v - lr * param.g

	def adagrad_update(self, param, lr):
		offset = 1e-9
		param.m += param.g * param.g 
		param.v = param.v - lr * param.g / np.sqrt(param.m + offset)
		

	def forward_once(self, x, h_pre, c_pre):
		# h_pre shape [h_size, 1]
		# x shape [x_size, 1]

		z = np.concatenate((h_pre, x), axis = 0)
		
		f = sigmoid(np.matmul(self.wf.v, z) + self.bf.v)
		i = sigmoid(np.matmul(self.wi.v, z) + self.bi.v)
		c1 = tanh(np.matmul(self.wc.v, z) + self.bc.v)
		c = f * c_pre + i * c1
		o = sigmoid(np.matmul(self.wo.v, z) + self.bo.v)

		h = o * tanh(c)
		p = np.matmul(self.w.v, h) + self.b.v
		y = np.exp(p) / np.sum(np.exp(p))
		return z, f, i, c1, c, o, h, p, y

	def backward_once(self, label, z, f, i, c1, c, o, h, p, y, d_h_nxt, d_c_nxt, c_pre):
		d_p = y.copy()
		d_p[label] -= 1
		d_h = d_h_nxt + np.matmul(self.w.v.T, d_p)
		d_o = d_h * tanh(c)
		d_o = d_o * o * (1 - o)
		d_c = d_c_nxt + d_h * o * (1 - tanh(c) * tanh(c))
		d_c1 = d_c * i
		d_c1 = d_c * (1 - c1 * c1)
		d_i = d_c * c1
		d_i = d_i * i * (1 - i)
		d_f = d_c * c_pre
		d_f = d_f * f * (1 - f)
		d_z = np.matmul(self.wo.v.T, d_o)
		d_z += np.matmul(self.wc.v.T, d_c1)
		d_z += np.matmul(self.wi.v.T, d_i)
		d_z += np.matmul(self.wf.v.T, d_f)
		d_c_pre = f * d_c
		d_h_pre = d_z[:self.h_size, :]


		self.w.g += np.matmul(d_p, h.T)
		self.b.g += d_p
		self.wc.g += np.matmul(d_c1, z.T)
		self.bc.g += d_c1
		self.wi.g += np.matmul(d_i, z.T)
		self.bi.g += d_i
		self.wf.g += np.matmul(d_f, z.T)
		self.bf.g += d_f
		self.wo.g += np.matmul(d_o, z.T)
		self.bo.g += d_o

		return d_c_pre, d_h_pre

	def forward(self, input, label):
		z_li, f_li, i_li, c1_li = [], [], [], []
		c_li, o_li, h_li, p_li, y_li = [], [], [], [], []

		sl = len(input)
		loss = 0
		for k in range(sl):
			x = np.zeros([self.x_size, 1])
			x[input[k], 0] = 1
			if k:
				h_pre = h_li[k-1].copy()
				c_pre = c_li[k-1].copy()
			else:
				h_pre = np.zeros([self.h_size, 1])
				c_pre = np.zeros([self.h_size, 1])

			z, f, i, c1, c, o, h, p, y = self.forward_once(x, h_pre, c_pre)
			z_li.append(z)
			f_li.append(f)
			i_li.append(i)
			c1_li.append(c1)
			c_li.append(c)
			o_li.append(o)
			h_li.append(h)
			p_li.append(p)
			y_li.append(y)

			loss -= np.log(y[label[k], 0])

		return loss / sl, z_li, f_li, i_li, c1_li, c_li, o_li, h_li, p_li, y_li

	def backward(self, input, label, z, f, i, c1, c, o, h, p, y):
		#z, f, i, c1, c, o, h, p, y, d_h_nxt, d_c_nxt, c_pre

		sl = len(input)
		d_h_nxt = np.zeros([self.h_size, 1])
		d_c_nxt = np.zeros([self.h_size, 1])
		for k in reversed(range(sl)):
			dh_nxt, dc_nxt = self.backward_once(label[k], z[k], f[k], i[k], c1[k], c[k], o[k], h[k], p[k], y[k],
												d_h_nxt, d_c_nxt, c[k-1])


	def sgd(self, lr):
		self.sgd_update(self.wf, lr)
		self.sgd_update(self.bf, lr)
		self.sgd_update(self.wi, lr)
		self.sgd_update(self.bi, lr)
		self.sgd_update(self.wc, lr)
		self.sgd_update(self.bc, lr)
		self.sgd_update(self.wo, lr)
		self.sgd_update(self.bo, lr)
		self.sgd_update(self.w, lr)
		self.sgd_update(self.b, lr)

	def adagrad(self, lr):
		self.adagrad_update(self.wf, lr)
		self.adagrad_update(self.bf, lr)
		self.adagrad_update(self.wi, lr)
		self.adagrad_update(self.bi, lr)
		self.adagrad_update(self.wc, lr)
		self.adagrad_update(self.bc, lr)
		self.adagrad_update(self.wo, lr)
		self.adagrad_update(self.bo, lr)
		self.adagrad_update(self.w, lr)
		self.adagrad_update(self.b, lr)

def gradient_check_wf(ck_time):
	#This is check for wf, for others also works
	lstm = LSTM(32, 128, 64)
	delta = 1e-5
	err_bound = 1e-5
	input = list(range(10))
	label = list(reversed(range(10)))
	for k in range(ck_time):
		ii = np.random.randint(0, lstm.h_size)
		jj = np.random.randint(0, lstm.x_size + lstm.h_size)
		lstm.zero_grad()
		loss, z, f, i, c1, c, o, h, p, y = lstm.forward(input, label)
		lstm.backward(input, label, z, f, i, c1, c, o, h, p, y)
		gradient = lstm.wc.g[ii, jj]
		lstm.wf.v[ii, jj] += delta
		loss_2, z, f, i, c1, c, o, h, p, y = lstm.forward(input, label)
		loss_delta = loss_2 - loss
		exp_delta = delta * gradient
		if np.abs(loss_delta - exp_delta) > err_bound:
			print("Wrong!")
			return
	print("Right!")
	
if __name__ == "__main__":
	gradient_check_wf(10)
