# -*- coding: utf-8 -*-
#%% Library

import math

# Neural Network
import torch
from torch import nn, Tensor
from torch.nn.utils import rnn

# Directory
import os
os.chdir('G:/其他電腦/MacBook Pro/Researches/_CODE_')
import tcn


#%%

class LSTM_Regr(nn.Module):  # 繼承自torch.nn.Module
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers_lstm: int,
        dropout: float,
        bidirectional: bool,
        stats_size: int ,
        last_hidden: int,
        act_fn: str,
        stablize: bool,
        bias_init = 90,
        set_target = 'bias',
        ) -> None:
        
        super(LSTM_Regr, self).__init__()  # 執行 Super Class (torch.nn.Module) 的 __init__()

        self.lstm = nn.LSTM( input_size = input_dim, 
                    hidden_size = hidden_size,
                    num_layers = num_layers_lstm,
                    dropout = dropout,
                    bidirectional = bidirectional,
                    batch_first=True)

        D = 2 if bidirectional == True else 1

        self.withStats = True if stats_size > 0 else False
        self.fc = nn.Linear( in_features = D * hidden_size + stats_size , out_features = last_hidden )
        
        self.stablize = stablize
        self.stablizer = nn.Tanh()
        
        if act_fn == 'Sigmoid':
            self.act = nn.Sigmoid()
        elif act_fn == 'Tanh':
            self.act = nn.Tanh()
        elif act_fn == 'ReLU':
            self.act = nn.ReLU()
        elif act_fn == 'ELU':
            self.act = nn.ELU()
        else:
            raise Exception("Unprogrammed Activation Function !")
        #self.act = nn.Tanh()
        self.out = nn.Linear( in_features = last_hidden , out_features = 1 )
        
        self.target = bias_init
        
        assert set_target in ['bias', 'add']
        self.set_target = set_target
        
        self.init_weights(self.target)
        
    def init_weights(self, bias_init, initrange = 0.1) -> None:
    
        if self.set_target == 'bias':
            self.out.bias.data.fill_(bias_init)
        
        #self.out.weight.data.uniform_(-initrange, initrange)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch, seq_lengths, statistics ): # -> Dict[str, torch.Tensor]:
        
        # statistics = [batch_size, num_stats_per_col = 7 , num_continuous_cols = 18]
        
        # batch = [batch_size, seq_length, data_dim]
        
        packed_padded = rnn.pack_padded_sequence(batch, seq_lengths, batch_first=True, enforce_sorted=False)

        lstm_output, (final_hidden_state , final_cell_state) = self.lstm(packed_padded)
        # lstm_output : [ batch_size, seq_len, num_directions * hid_size ]
        # final_hidden_state : [ num_layers * num_directions, batch_size,  hid_size ]
        
        #concat the final forward and backward hidden state
        if self.withStats:
            if self.stablize:
                stats_flattened = self.stablizer(statistics.view(statistics.size()[0], -1 ))
            else:
                stats_flattened = statistics.view(statistics.size()[0], -1 )
            
            hidden = torch.cat((final_hidden_state[-2,:,:], final_hidden_state[-1,:,:], stats_flattened), dim = 1)
        else:
            hidden = torch.cat((final_hidden_state[-2,:,:], final_hidden_state[-1,:,:]), dim = 1)
        #hidden : [batch size, hid_size * num_directions]
        
        #hidden = self.stablize(hidden)
        
        dense=   self.fc(hidden)  

        #Final activation function
        act =  self.stablizer( self.act(dense) ) if self.stablize else self.act(dense)
        
        output = self.out(act)
        
        if self.set_target == 'add':
            output = torch.add(output , self.target)
        
        return output
    
#%%

class GRU_Regr(nn.Module):  # 繼承自torch.nn.Module
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers_gru: int,
        dropout: float,
        bidirectional: bool,
        stats_size: int ,
        last_hidden: int,
        act_fn: str,
        stablize: bool,
        bias_init = 90,
        set_target = 'bias',
        ) -> None:
        
        super(GRU_Regr, self).__init__()  # 執行 Super Class (torch.nn.Module) 的 __init__()
        
        self.gru = nn.GRU( input_size = input_dim, 
                    hidden_size = hidden_size,
                    num_layers = num_layers_gru,
                    dropout = dropout,
                    bidirectional = bidirectional,
                    batch_first=True)
        

        D = 2 if bidirectional == True else 1

        self.withStats = True if stats_size > 0 else False
        self.fc = nn.Linear( in_features = D * hidden_size + stats_size , out_features = last_hidden )
        
        self.stablize = stablize
        self.stablizer = nn.Tanh()
        
        if act_fn == 'Sigmoid':
            self.act = nn.Sigmoid()
        elif act_fn == 'Tanh':
            self.act = nn.Tanh()
        elif act_fn == 'ReLU':
            self.act = nn.ReLU()
        elif act_fn == 'ELU':
            self.act = nn.ELU()
        else:
            raise Exception("Unprogrammed Activation Function !")
        #self.act = nn.Tanh()
        self.out = nn.Linear( in_features = last_hidden , out_features = 1 )
        
        self.target = bias_init
        
        assert set_target in ['bias', 'add']
        self.set_target = set_target
        
        self.init_weights(self.target)
        
    def init_weights(self, bias_init, initrange = 0.1) -> None:
    
        if self.set_target == 'bias':
            self.out.bias.data.fill_(bias_init)
        #self.out.weight.data.uniform_(-initrange, initrange)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch, seq_lengths, statistics ): # -> Dict[str, torch.Tensor]:
        
        # statistics = [batch_size, num_stats_per_col = 7 , num_continuous_cols = 18]
        
        # batch = [batch_size, seq_length, data_dim]
        
        packed_padded = rnn.pack_padded_sequence(batch, seq_lengths, batch_first=True, enforce_sorted=False)

        gru_output, final_hidden_state = self.gru(packed_padded)
        # lstm_output : [ batch_size, seq_len, num_directions * hid_size ]
        # final_hidden_state : [ num_layers * num_directions, batch_size,  hid_size ]
        
        #concat the final forward and backward hidden state
        if self.withStats:
            if self.stablize:
                stats_flattened = self.stablizer(statistics.view(statistics.size()[0], -1 ))
            else:
                stats_flattened = statistics.view(statistics.size()[0], -1 )
            
            hidden = torch.cat((final_hidden_state[-2,:,:], final_hidden_state[-1,:,:], stats_flattened), dim = 1)
        else:
            hidden = torch.cat((final_hidden_state[-2,:,:], final_hidden_state[-1,:,:]), dim = 1)
        #hidden : [batch size, hid_size * num_directions]
        
        #hidden = self.stablize(hidden)
        
        dense=   self.fc(hidden)  

        #Final activation function
        act =  self.stablizer( self.act(dense) ) if self.stablize else self.act(dense)
        
        output = self.out(act)
        
        if self.set_target == 'add':
            output = torch.add(output , self.target)
        
        return output
    
#%%

class TCN_Regr(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int,
        num_levels_tcn: int,
        kernel_size: int, 
        dropout: float,
        stats_size: int ,
        last_hidden: int,
        act_fn: str,
        stablize: bool,
        bias_init = 90,
        set_target = 'bias',
        ):
        
        super(TCN_Regr, self).__init__()
        
        num_channels =  [hidden_size] * num_levels_tcn
        
        self.tcn = tcn.TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.withStats = True if stats_size > 0 else False
        self.fc = nn.Linear( in_features = num_channels[-1] + stats_size , out_features = last_hidden )
        
        self.stablize = stablize
        self.stablizer = nn.Tanh()
        
        if act_fn == 'Sigmoid':
            self.act = nn.Sigmoid()
        elif act_fn == 'Tanh':
            self.act = nn.Tanh()
        elif act_fn == 'ReLU':
            self.act = nn.ReLU()
        elif act_fn == 'ELU':
            self.act = nn.ELU()
        else:
            raise Exception("Unprogrammed Activation Function !")
        #self.act = nn.Tanh()
        self.out = nn.Linear( in_features = last_hidden , out_features = 1 )
        
        self.target = bias_init
        
        assert set_target in ['bias', 'add']
        self.set_target = set_target
        
        self.init_weights(self.target)
        
    def init_weights(self, bias_init, initrange = 0.1) -> None:
        if self.set_target == 'bias':
            self.out.bias.data.fill_(bias_init)
        self.fc.weight.data.normal_(0, 0.01)

    def forward(self, batch, seq_lengths, statistics ):
        # statistics = [batch_size, num_stats_per_var, num_vars]
        # batch = [seq_length, batch_size, data_dim] or [batch_size, seq_length, data_dim] if batch_first while padding
        
        batch = batch.flip(1) # batch first
        
        batch = torch.transpose(batch, 1, 2)
        
        #packed_padded = rnn.pack_padded_sequence(batch, seq_lengths, batch_first=True, enforce_sorted=False)
        
        tcn_out = self.tcn(batch)    # In/Out size: batch_size, num_channels, seq_len
        tcn_out = tcn_out[:, :, -1]

        
        if self.withStats:
            if self.stablize:
                stats_flattened = self.stablizer(statistics.view(statistics.size()[0], -1 ))
            else:
                stats_flattened = statistics.view(statistics.size()[0], -1 )
            
            hidden = torch.cat((tcn_out, stats_flattened), dim = 1)
        else:
            hidden =tcn_out
        #hidden : [batch size, hid_size * num_directions]
        
        #hidden = self.stablize(hidden)

        dense=   self.fc(hidden)  

        #Final activation function
        act =  self.stablizer( self.act(dense) ) if self.stablize else self.act(dense)
        
        output = self.out(act)
        
        if self.set_target == 'add':
            output = torch.add(output , self.target)
        
        return output
        

#%%

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float =0.1, max_len: int = 14678):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model%2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:,0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
    
class Transformer_Regr(nn.Module):  # 繼承自torch.nn.Module
    def __init__(
        self,
        d_model: int,   # d_model: dimension of data (embedded)
        num_encoder_layers: int,
        nhead: int,           
        dim_feedforward: int,
        dim_reduction: int,
        dropout: float,
        stats_size: int ,
        last_hidden: int,
        act_fn: str,
        stablize: bool,
        bias_init = 90,
        set_target = 'bias',
        pos_encoding = False,
        ):
        
        super().__init__()  # 執行 Super Class (torch.nn.Module) 的 __init__()
        self.pos_encoding = pos_encoding
        #if pos_encoding:
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        #self.pos_encoder = positional_encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.d_model = d_model
        
        self.dim_reduction = nn.Linear(in_features= 340, out_features = dim_reduction)

        self.withStats = True if stats_size > 0 else False
        self.fc = nn.Linear( in_features = d_model * dim_reduction + stats_size , out_features = last_hidden )
        
        self.stablize = stablize
        self.stablizer = nn.Tanh()
        
        if act_fn == 'Sigmoid':
            self.act = nn.Sigmoid()
        elif act_fn == 'Tanh':
            self.act = nn.Tanh()
        elif act_fn == 'ReLU':
            self.act = nn.ReLU()
        elif act_fn == 'ELU':
            self.act = nn.ELU()
        else:
            raise Exception("Unprogrammed Activation Function !")
        #self.act = nn.Tanh()
        self.out = nn.Linear( in_features = last_hidden , out_features = 1 )
        
        self.target = bias_init
        
        assert set_target in ['bias', 'add']
        self.set_target = set_target
        
        self.init_weights(self.target)
        
    def init_weights(self, bias_init, initrange = 0.1) -> None:
        
        if self.set_target == 'bias':
            self.out.bias.data.fill_(bias_init)
        
        self.dim_reduction.bias.data.zero_() 
        self.dim_reduction.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, seq_len = None, statistics = None, src_mask: Tensor = None) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        batch_size = src.size()[1]
        # statistics = [batch_size, num_stats_per_col = 7 , num_continuous_cols = 18]
        # batch = [seq_length, batch_size, data_dim] or [batch_size, seq_length, data_dim] if batch_first while padding
        
        if self.pos_encoding:
            src = self.pos_encoder(src)
            
        #output = self.transformer_encoder(src, src_mask)
        # TODO: create 'src_mask' (attention focus )
        # TODO: create 'src_key_padding_mask' (transformer's version of packing -> attention=0 )
        
        encoded = self.transformer_encoder(src)
        # encoded : [padded_to_len, batch_size, d_model]
        encoded = torch.transpose(encoded, 0, 2)
        
        reduced = self.dim_reduction(encoded)
        # reduced = [ d_model, batch_size, 2 ]
        reduced = torch.transpose(reduced, 0, 2)
        reduced = torch.reshape(reduced, (batch_size, -1))
        # reduced = [ batch_size, d_model*2 ]
        
        #concat the final forward and backward hidden state
        if self.withStats:
            if self.stablize:
                stats_flattened = self.stablizer(statistics.view(statistics.size()[0], -1 ))
            else:
                stats_flattened = statistics.view(statistics.size()[0], -1 )
            
            #concat the final forward and backward hidden state
            hidden = torch.cat((reduced, stats_flattened), dim = 1)
            #hidden : [batch size, hid_size_lstm * num_directions]   
        else:
            hidden = reduced
        #hidden : [batch size, hid_size * num_directions]
        
        #hidden = self.stablize(hidden)
        
        dense=   self.fc(hidden)  

        #Final activation function
        act =  self.stablizer( self.act(dense) ) if self.stablize else self.act(dense)
        
        output = self.out(act)
        
        if self.set_target == 'add':
            output = torch.add(output , self.target)
        
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
