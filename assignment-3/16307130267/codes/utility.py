import json
import numpy as np

def clear_progressbar():
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")

def load_params(jsonFile):
    with open(jsonFile) as f:
        return json.load(f)

def perplexity(input_data):
	from loss import softmax
	N, T, V = input_data.shape
	p = 1
	for i in range(N):
		p *= softmax(input_data[i])
	return pow(1/p, 1/N) / N