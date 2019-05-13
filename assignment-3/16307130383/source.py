import json
import string
import random
import sys
sys.path.append('../')

data = []
with open('../poet.tang.20000.json', encoding='utf-8') as json_file:
  data += json.load(json_file)
with open('../poet.tang.21000.json', encoding='utf-8') as json_file:
  data += json.load(json_file)
with open('../poet.tang.22000.json', encoding='utf-8') as json_file:
  data += json.load(json_file)
with open('../poet.tang.23000.json', encoding='utf-8') as json_file:
  data += json.load(json_file)
with open('../poet.tang.24000.json', encoding='utf-8') as json_file:
  data += json.load(json_file)

paragraphs = [item['paragraphs'] for item in data]
random.shuffle(paragraphs)

def get_vocab_seqs(sample_size=1000, seq_lenth=48):
  assert sample_size <= 4000

  vocab = {}
  v_count = 0
  for paragraph in paragraphs[:sample_size]:
    for sentence in paragraph:
      for i, word in enumerate(sentence):
        if word in vocab:
          pass
        else:
          vocab[word] = v_count
          v_count += 1

  vocab['EOS'] = v_count
  v_count += 1
  vocab['OOV'] = v_count
  v_count += 1

  sequences = []
  for paragraph in paragraphs[:sample_size]:
    seq = ""
    for sentence in paragraph:
      seq += sentence
      if len(seq) >= seq_lenth:
        sequences.append(seq[:seq_lenth])
        seq = ""
  return vocab, sequences

def embedding(vocab, sequences):
  xs = []
  ys = []
  for sequence in sequences:
    seq = []
    for word in sequence:
      if word in vocab:
        seq.append(vocab[word])
      else:
        seq.append(vocab['OOV'])
    seq_next = seq[1:]
    seq_next.append(vocab['EOS'])
    xs.append(seq)
    ys.append(seq_next)
  return xs, ys