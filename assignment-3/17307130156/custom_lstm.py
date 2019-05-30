
import math

import torch
import torch.nn as nn
import torch.nn.init as init

from torch.nn import Parameter
from torch.nn.init import xavier_uniform
from torch.nn.utils.rnn import PackedSequence, get_packed_sequence

from torch import Tensor
from collections import namedtuple


def LSTM(input_size, hidden_size, num_layers, batch_first=False, bidirectional=False):
    if bidirectional:
        return BidirLSTMLayer(input_size, hidden_size)
    else:
        return LSTMLayer(input_size, hidden_size)
    

class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):

        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Parameter Wrapper: enable autograd and add to the parameter list of this module, which can be refered as parameters()
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))

        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))
        # Initialize biases for LSTM’s forget gate to 1 to remember more by default.
        # self.reset_parameters()

        self.bias_ih[hidden_size: 2 * hidden_size] = 0
        self.bias_hh[hidden_size: 2 * hidden_size] = 1

        self.bias_ih = Parameter(self.bias_ih)
        self.bias_hh = Parameter(self.bias_hh)


    # Do matters!
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, input, state):
        '''
        type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        :input: (batch_size, input_size)
        :state: ((batch_size, hidden_size), (batch_size, hidden_size))
        '''
        hx, cx = state

        # '+' is broadcastable, while torch.mm not
        '''
        [   Wi 
            Wf       * z + b 
            Wg 
            Wo  ]
        '''
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        # Thus gates is of shape (batch_size, 4 * hidden_size)

        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)
        # Each of shape (batch_size, hidden_size)

        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)

        # '*' element-wise multiplication
        cy = (forget_gate * cx) + (in_gate * cell_gate)
        hy = out_gate * torch.tanh(cy)

        return hy, (hy, cy)

class ReverseLSTMLayer(nn.Module):

    def __init__(self, *cell_args):

        super(ReverseLSTMLayer, self).__init__()
        self.cell = LSTMCell(*cell_args)

    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        
        inputs = torch.unbind(input, 0)[::-1] # Reverse
        outputs = []

        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]

        return torch.stack(reverse(outputs)), state


class LSTMLayer(nn.Module):

    def __init__(self, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = LSTMCell(*cell_args)


    def forward(self, input, state):
        '''
        type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        input is of shape (seq_size, batch_size, feature)
        state is of shape ((1, batch_size, hidden_size), (1, batch_size, hidden_size))
        '''
        state = state[0].squeeze(0), state[1].squeeze(0)

        # TODO Support batch first
        seq_dim = 0

        inputs = input.unbind(seq_dim) # Split by dimension 0 and return a tuple, it's the reverse operation of torch.stack()
        outputs = []

        seq_size = len(inputs)
        for i in range(seq_size):
            # All i-th character in batches 
            # print (state[0].shape, state[1].shape)
            out, state = self.cell(inputs[i], state)

            # TODO Mermory efficient
            # out, state = self.cell(inputs[i][: batch_size], state[: batch_size])
            # That's why we need sorted sequence 

            outputs += [out]

        #TODO Right? 
        state = state[0].unsqueeze(0), state[1].unsqueeze(0)
        # torch.stack(seq, dim=0, out=None) -> Tensor : Concatenates sequence of tensors along a new dimension.
        # At this point, it just turn List into Tensor
        # I.e. torch.stack(outputs) is of shape (seq_size, batch_size, feature), which is the same as input
        return torch.stack(outputs), state




class BidirLSTMLayer(nn.Module):
    __constants__ = ['directions']

    def __init__(self, *cell_args):
        super(BidirLSTMLayer, self).__init__()
        self.directions = nn.ModuleList([
            LSTMLayer(*cell_args),
            ReverseLSTMLayer(*cell_args),
        ])

    def forward(self, input, states):
        # type: (Tensor, List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
        # List[LSTMState]: [forward LSTMState, backward LSTMState]
        outputs = []
        output_states = []
        i = 0
        for direction in self.directions:
            state = states[i]
            out, out_state = direction(input, state)
            outputs += [out]
            output_states += [out_state]
            i += 1
        return torch.cat(outputs, -1), output_states



LSTMState = namedtuple('LSTMState', ['hx', 'cx'])

def test(seq_len, batch, input_size, hidden_size):

    inp = torch.randn(seq_len, batch, input_size)
    state = LSTMState(hx=torch.randn(batch, hidden_size), cx=torch.randn(batch, hidden_size))

    custom_lstm = LSTMLayer(input_size, hidden_size)
    custom_out, custom_out_state = custom_lstm(inp, state)

    # Pytorch native LSTM
    native_lstm = nn.LSTM(input_size, hidden_size, 1)
    native_state = LSTMState(hx=state.hx.unsqueeze(0), cx=state.cx.unsqueeze(0)) # Native LSTM expects all input to be 3-D Tensor, including state, while our custom LSTM only need 2-D Tensor for state

    for native_param, custom_param in zip(native_lstm.all_weights[0], custom_lstm.parameters()):
        assert native_param.shape == custom_param.shape
        with torch.no_grad():
            native_param.copy_(custom_param)

    native_out, native_out_state = native_lstm(inp, native_state)
    
    # Testing correctness of our custom lstm
    assert (custom_out - native_out).abs().max() < 1e-5
    assert (custom_out_state[0] - native_out_state[0]).abs().max() < 1e-5
    assert (custom_out_state[1] - native_out_state[1]).abs().max() < 1e-5





if __name__ == '__main__':

    test(10, 3, 10, 2)

