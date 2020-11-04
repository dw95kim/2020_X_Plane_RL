import numpy as np
import torch
from torch import nn

import os
import random

####################################################
# Seq2Seq LSTM AutoEncoder Model
# 	- predict locations
####################################################

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_dim, num_layers, isCuda=False):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.isCuda = isCuda
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True)
        
    def forward(self, input):
        output, hidden = self.lstm(input)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, hidden_dim, num_layers, dropout=0.5, isCuda=False):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.isCuda = isCuda
        self.lstm = nn.LSTM(hidden_size, hidden_dim, num_layers, batch_first=True)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_dim, output_size)
        self.tanh = nn.Tanh()

    def forward(self, encoded_input, hidden):
        decoded_output, hidden = self.lstm(encoded_input, hidden)
        decoded_output = self.dropout(decoded_output)
        decoded_output = self.linear(decoded_output)
        return decoded_output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_dim, num_layers, dropout=0.5, isCuda=False):
        super(Seq2Seq, self).__init__()
        self.isCuda = isCuda
        self.encoder = EncoderRNN(input_size, hidden_size, hidden_dim, num_layers, isCuda)
        self.decoder = DecoderRNN(hidden_size, hidden_size, hidden_dim, num_layers, dropout, isCuda)

    def forward(self, in_data, last_location, pred_length, device, teacher_forcing_ratio=0, teacher_location=None):
        batch_size = in_data.shape[0]
        in_data = in_data.to(device)
        out_dim = self.decoder.output_size
        self.pred_length = pred_length
        outputs = torch.zeros(batch_size, self.pred_length, out_dim)
        if self.isCuda:
            outputs = outputs.cuda()
        
        encoded_output, hidden = self.encoder(in_data)
        decoder_input = last_location
        for t in range(self.pred_length):
            now_out, hidden = self.decoder(decoder_input, hidden)
            now_out += decoder_input
            outputs[:,t:t+1] = now_out 
            teacher_forcing = False
            decoder_input = (teacher_location[:,t:t+1] if (type(teacher_location) is not type(None)) and teacher_force else now_out)
        return outputs

class Enemy_Predict():
    def __init__(self, model_path, skip_data_cnt):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.future_frames = 10
        self.half_data_cnt = skip_data_cnt * future_frames

        self.scale = 2000
        self.raw_input = []
        self.input = []
        self.model = Seq2Seq(input_size=(2), hidden_size=2, hidden_dim = 512, num_layers=2, dropout=0.5, isCuda=False).to(self.device)

        self.last_position = 0
        self.zero_pos = 0

        self.model.load_state_dict(torch.load(model_path), , map_location=torch.device('cpu'))
        print('Successfull loaded from {}'.format(model_path))
        self.model.eval()

    def set_input(self, input_data):
        assert (len(input_data[0]) == self.future_frames), "the number of input length is not future frames"
        assert (type(input_data) == list), "input type is not list"

        self.raw_input = torch.tensor(input_data).to(self.device)
        self.scaling_input = self.raw_input * self.scale
        self.last_position = self.scaling_input[:, -1:, :].to(self.device)

        self.input = self.scaling_input - self.last_position
        self.zero_pos = self.input[:, -1:, :].to(self.device)

    def run(self):
        output = self.model(self.input.float(), self.zero_pos.float(), self.future_frames, self.device)
        return (output + self.last_position) / self.scale
