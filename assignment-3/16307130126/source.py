import torch
from torch import nn
from torch import optim
import numpy as np
import os
import random
import re
import json
import matplotlib.pyplot as plt


class LSTM_without_torch(nn.Module):
	def __init__(self, vocab_size):
		super(LSTM_without_torch, self).__init__()
		self.input_size = 128
		self.hidden_size = 256
		# self.num_layers = 1
		self.output_size = vocab_size
		self.embeddings = nn.Embedding(vocab_size, self.input_size)

		self.Wf = nn.Parameter((2 * torch.rand(self.input_size + self.hidden_size, self.hidden_size) - 1) / np.sqrt(
			self.input_size + self.hidden_size))
		self.Wi = nn.Parameter((2 * torch.rand(self.input_size + self.hidden_size, self.hidden_size) - 1) / np.sqrt(
			self.input_size + self.hidden_size))
		self.Wc = nn.Parameter((2 * torch.rand(self.input_size + self.hidden_size, self.hidden_size) - 1) / np.sqrt(
			self.input_size + self.hidden_size))
		self.Wo = nn.Parameter((2 * torch.rand(self.input_size + self.hidden_size, self.hidden_size) - 1) / np.sqrt(
			self.input_size + self.hidden_size))

		self.bf = nn.Parameter((2 * torch.rand(self.hidden_size) - 1) / np.sqrt(self.hidden_size))
		self.bi = nn.Parameter((2 * torch.rand(self.hidden_size) - 1) / np.sqrt(self.hidden_size))
		self.bc = nn.Parameter((2 * torch.rand(self.hidden_size) - 1) / np.sqrt(self.hidden_size))
		self.bo = nn.Parameter((2 * torch.rand(self.hidden_size) - 1) / np.sqrt(self.hidden_size))
		self.W = nn.Parameter((2 * torch.rand(self.hidden_size, vocab_size) - 1) / np.sqrt(self.hidden_size))
		self.b = nn.Parameter((2 * torch.rand(vocab_size) - 1) / np.sqrt(vocab_size))

	def forward(self, input, hidden=None):

		batch_size = input.size()[0]
		seq_len = input.size()[1]
		if not hidden:
			h0 = torch.zeros(batch_size, self.hidden_size)
			c0 = torch.zeros(batch_size, self.hidden_size)
		else:
			h0 = hidden[0]
			c0 = hidden[1]
		Ctt = c0
		ht = h0
		at = torch.FloatTensor(seq_len, batch_size, self.output_size)
		embeds = self.embeddings(input.transpose(0, 1))
		for i in range(seq_len):
			z = torch.cat(tensors=(embeds[i], ht), dim=1)
			ft = torch.sigmoid(z.mm(self.Wf) + self.bf)
			it = torch.sigmoid(z.mm(self.Wi) + self.bi)
			b_Ct = torch.tanh(z.mm(self.Wc) + self.bc)
			Ct = ft * Ctt + it * b_Ct
			ot = torch.sigmoid(z.mm(self.Wo) + self.bo)
			ht = ot * torch.tanh(Ct)
			at[i] = ht.mm(self.W) + self.b
			Ctt = Ct
		return at, (ht, Ct)


def get_data():
	print('read tangshi.')
	file = open('tangshi_more.txt', 'r', encoding='UTF-8')
	# file = open('test.txt', 'r')
	global data, max_len
	data = []
	try:
		s = ""
		t = file.readline()
		# print(t)
		while True:
			if not t:
				break
			t = t[:-1]
			# print(t)
			if len(t) == 0:
				#ss = re.findall(r'.{64}', s)
				ss = s.split('。')
				tmp = ''
				for step, st in enumerate(ss):
					if (len(st) > 1):
						if (len(tmp) + len(st) + 1<= 64):
							tmp += st + '。'
						else:
							if (len(tmp)>=48):
								#print(tmp)
								data.append(tmp)
							tmp = st + '。'
				if (len(tmp) >= 48):
					data.append(tmp)
					tmp = ""
				if (len(data) >= 700):
					break
				s = ""
			else:
				s += t
			t = file.readline()
	finally:
		file.close()

	max_len = 0
	for x in data: max_len = max(max_len, len(x))
	for i in range(len(data)):
		data[i] += 'E' * (max_len - len(data[i]))
	print('data =',data)
	print("poem max len=", max_len)
	print("poem size=", len(data))
	print("read data end.")


def preprocess():
	print('change words to integer.')
	global data, vocab, vocab_lt, Rdata, max_len
	vocab = {}
	vocab_lt = []
	tot = 0
	for poem in data:
		for ch in poem:
			if ch not in vocab:
				vocab[ch] = tot
				tot += 1
				vocab_lt.append(ch)
	vocab['OOV'] = tot
	tot += 1
	vocab_lt.append('OOV')
	print(vocab)
	print(vocab_lt)
	# print("poem size=",len(data))

	lt = ['日', '紅', '山', '夜', '湖', '海', '月']
	for x in lt:
		if x not in vocab:
			print("WARNING")
			exit(0)

	print('vocabulary size=', tot)
	lt = list(map(lambda x: list(map(lambda y: vocab[y], x)), data))
	Rdata = torch.LongTensor(lt)
	print(Rdata)


def generate(model, word):
	global vocab, vocab_lt
	poem_len = 1
	poem = [word]
	input = torch.LongTensor([vocab[word]]).view(1, 1)
	hidden = None
	v = None
	for i in range(max_len - 1):
		output, hidden = model(input, hidden)
		f = nn.Softmax()
		tmp = f(output[0][0])
		v = int(torch.argmax(tmp).numpy())
		#print("v=", v)

		poem.append(vocab_lt[v])
		input = torch.LongTensor([v]).view(1, 1)
	print(poem)
	s = ""
	for word in poem:
		if word != 'E':
			s += word
		else:
			break
			#s += '\n'
	print(s)


def calc_perplexity(model):
	global test_data, max_len
	ans = np.zeros(len(test_data))

	y, _ = model(test_data)
	y = y.transpose(0, 1)
	y = torch.softmax(y, 0).detach().numpy()
	for i in range(len(y)):
		tmp = 1
		for j in range(max_len - 1):
			tmp *= np.power(1 / y[i][j][test_data[i][j + 1]], 1 / (max_len - 1))
		ans[i] = tmp
	ret = np.average(ans)
	return ret


def train():
	print('training...')
	batch_size = 20
	global Rdata, test_data, vocab, max_len, vocab_lt
	vocab_size = len(vocab)
	train_data = Rdata[:int(0.8 * len(Rdata)+1)]
	# train_data = Rdata
	test_data = Rdata[int(0.8 * len(Rdata)) + 2:]
	lstm = LSTM_without_torch(len(vocab))
	if os.path.exists('lstm_model_s.pth'):
		print('load model...')
		lstm.load_state_dict(torch.load('lstm_model_s.pth'))
	batches = [train_data[k:k + batch_size] for k in range(0, len(train_data), batch_size)]
	random.shuffle(batches)
	#optimizer = optim.SGD(lstm.parameters(), lr=0.5,momentum=0.9)
	optimizer = optim.Adam(lstm.parameters(),lr=0.01)
	loss_function = nn.CrossEntropyLoss()
	cnt = 0
	loss_epoch = []
	perplexity_epoch = []
	min_per = 100000
	cnt = 0
	try:
		f = open('Adam.txt', 'r')
		s = f.read()
		s = s.replace("'", '"')
		dt = json.loads(s)
		loss_epoch = dt['loss']
		perplexity_epoch = dt['perplexity']
		for x in perplexity_epoch:
			if x < min_per:
				cnt = 0
				min_per = x
			else:
				cnt += 1


		f.close()
	except Exception:
		pass

	#print('cnt=',cnt)
	try:
		epoch = 0
		while True:
			loss_step = np.zeros(len(batches))
			for step, data in enumerate(batches):
				input, target = data[:, :-1], data[:, 1:].t().contiguous()
				y, _ = lstm(input)
				loss = loss_function(y.view(-1, vocab_size), target.view(-1))
				loss_step[step] = loss
				optimizer.zero_grad()
				print('step ', step, ' loss=', loss.detach().numpy())
				loss.backward()
				optimizer.step()
			epoch += 1
			loss_avg = np.average(loss_step)
			loss_epoch.append(loss_avg)
			perplexity = calc_perplexity(lstm)
			print('epoch %d loss %f' % (epoch, loss_avg))
			print('epoch %d perplexity %f' % (epoch, perplexity))
			perplexity_epoch.append(perplexity)
			if perplexity > min_per:
				cnt += 1
			else:
				min_per = perplexity
				cnt = 0
			if cnt >= 10:
				print(cnt)
				break
	except KeyboardInterrupt as e:
		pass
	print('save model...')
	torch.save(lstm.state_dict(), 'lstm_model_s.pth')
	f = open('Adam.txt', 'w')
	dt = {'loss': loss_epoch, 'perplexity': perplexity_epoch}
	f.write(str(dt))
	f.close()
	lt = ['日', '紅', '山', '夜', '湖', '海', '月']
	for x in lt:
		generate(lstm, x)


# generate(lstm, '巴')

def draw():
	f1 = open('Adam.txt', 'r')
	f2 = open('SGD.txt','r')
	f3 = open('SGD_momentun.txt', 'r')

	s = f1.read().replace("'", '"')
	f1.close()
	dt1 = json.loads(s)
	s = f2.read().replace("'", '"')
	f1.close()
	dt2 = json.loads(s)
	s = f3.read().replace("'", '"')
	f1.close()
	dt3 = json.loads(s)

	Adam_loss = dt1['loss']
	Adam_per = dt1['perplexity']
	SGD_loss = dt2['loss']
	SGD_per = dt2['loss']
	SGD_m_loss = dt3['loss']
	SGD_m_per = dt3['loss']
	max_len = max(len(Adam_loss),len(SGD_m_loss),len(SGD_loss))
	time = np.arange(len(Adam_loss))
	x = max_len - len(Adam_loss)
	for i in range(x):
		Adam_loss.append(None)
	x = max_len - len(SGD_m_loss)
	for i in range(x):
		SGD_m_loss.append(None)
	x = max_len - len(SGD_loss)
	for i in range(x):
		SGD_loss.append(None)
	plt.plot(time, Adam_loss)
	plt.plot(time, SGD_m_loss)
	plt.plot(time, SGD_loss)
	plt.legend(["Adam loss", "SGD with momentun loss","SGD loss"])
	plt.show()

if __name__ == "__main__":
	np.set_printoptions(threshold=np.inf)
	get_data()
	preprocess()
	train()
	#draw()