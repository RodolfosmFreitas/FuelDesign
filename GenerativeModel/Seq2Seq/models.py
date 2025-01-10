# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 07:02:10 2024

@author: Rodolfo Freitas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    """Encoder Layer for FuelToFuel Model.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_layers: int,
                 embedding_size: int,
                 dropout_p: float = 0.1,
                 **kwargs):
        """Initialize the EncoderRNN layer.

        Parameters
        ----------
        input_size: int
            The number of expected features.
        hidden_size: int
            The number of features in the hidden state.
        embedding_size: int
            Latent space dimension (fingerprint)
        n_layers: int
            The number of recurrent layers  
        dropout_p: float (default 0.1)
            The dropout probability to use during training.

        """
        super(EncoderRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p, batch_first=True)
        
        # Fingerprint extraction layer
        self.finger = nn.Linear(2*hidden_size, embedding_size)
        
        # Initialize the Net
        self.initialize_weights()
    
    def forward(self, input: torch.Tensor):
        """Returns Embeddings according to provided sequences.

        Parameters
        ----------
        input: torch.Tensor
            Batch of input sequences.

        Returns
        -------
        output: torch.Tensor
            Batch of Embeddings.
        hidden: torch.Tensor
            Batch of hidden states.

        """
        embedded = self.dropout(self.embedding(input))
        # embeddiing_shape: (N, 1, hidden_size)

        output, hidden = self.rnn(embedded)
        # outputs shape: (N, seq_length, hidden_size)
        
        # Fingerprint extraction layer
        z = F.relu(self.finger(torch.cat((hidden[0], hidden[-1]),dim=-1)))
        
        return output, z
    
    # Initialize network weights and biases using Xavier initialization
    
    def initialize_weights(self):
        for m in self.modules():
           
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    # count the number of parameters
    def num_parameters(self):
        n_params = 0
        for name, param in self.named_parameters():
            n_params += param.numel()
        return n_params
    
'''
ATTENTION DECODER
'''

class BahdanauAttention(nn.Module):
    '''
    Neural Machine Translation by Jointly Learning to Align and Translate
    Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
    Published in International Conference on… 1 September 2014
    '''
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ua = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Va = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, keys, values):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights
    
class LuongAttention(nn.Module):
    '''
    Thang Luong, Hieu Pham, and Christopher D. Manning. 2015. 
    Effective Approaches to Attention-based Neural Machine Translation. 
    In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, 
    pages 1412–1421, Lisbon, Portugal. Association for Computational Linguistics.
    '''
    def __init__(self, hidden_size):
        super(LuongAttention, self).__init__()
        self.Wa = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.Va = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, keys, values):
        scores = self.Va(torch.tanh(self.Wa(torch.cat((query, keys),2))))
        scores = scores.squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights
    
class GeneralAttention(nn.Module):
    def __init__(self, hidden_size):
        super(GeneralAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, keys, values):
        scores = self.Wa(query)
        scores = scores.squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights
    
class MultiHeadAttention(nn.Module):
    '''
    Attention Is All You Need
    Ashish Vaswani et al. Advances in Neural Information Processing Systems 30 (NIPS 2017)
    '''

    def __init__(self, hidden_size, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(hidden_size, n_heads, bias=False, batch_first=True)

    def forward(self, query, keys, values):
        query = query[:,-1,:].unsqueeze(1)
        context, weights = self.multihead_attn(query, keys, values)
        return context, weights

class AttnDecoderRNN(nn.Module):
    """Decoder Layer for FuelToFuel Model.
    """

    def __init__(self,
                 hidden_size: int,
                 output_size: int,
                 n_layers: int,
                 max_length: int,
                 embedding_size: int,
                 dropout_p: float = 0.1,
                 Att_method: str = 'Bahdanau',
                 **kwargs):
        """Initialize the DecoderRNN layer.

        Parameters
        ----------
        hidden_size: int
            Number of features in the hidden state.
        output_size: int
            Number of expected features.
        max_length: int
            Maximum length of the sequence.
        batch_size: int
            Batch size of the input.
        dropout_p: float (default 0.1)
            The dropout probability to use during training.
        """
        super(AttnDecoderRNN, self).__init__(**kwargs)
        self.num_layers = n_layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        # Attention Method to use
        if Att_method == 'MultiHead':
            self.attention = MultiHeadAttention(hidden_size, n_heads=8)
        elif Att_method == 'Bahdanau':
            self.attention = BahdanauAttention(hidden_size)
        elif Att_method == 'Luong':
            self.attention = LuongAttention(hidden_size)
        elif Att_method == 'General':
            self.attention = GeneralAttention(hidden_size)
        
        self.dropout = nn.Dropout(dropout_p)
        
        # state connection layer
        self.connection = nn.Linear(embedding_size, hidden_size)
        
        self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p, batch_first=True)
        self.out = nn.Linear(2*hidden_size, output_size)
        self.MAX_LENGTH = max_length
        
        # Initialize the Net
        self.initialize_weights()

    
    def forward(self, encoder_outputs, fingerprints, target_tensor=None):
        """
        Parameters
        ----------
        inputs: List[torch.Tensor]
            A list of tensor containg encoder_hidden and target_tensor.

        Returns
        -------
        decoder_outputs: torch.Tensor
            Predicted output sequences.
        decoder_hidden: torch.Tensor
            Hidden state of the decoder.

        """
        
        batch_size = encoder_outputs.shape[0]
        decoder_input = torch.zeros(batch_size,
                                   1,
                                   dtype=torch.long,
                                   device=fingerprints.device)
        
        # laten space -> Interpreter
        z = F.relu(self.connection(fingerprints))
        decoder_hidden = torch.stack(self.num_layers * [z])
        
        decoder_outputs = []
        attentions = []

        for i in range(self.MAX_LENGTH):
            
            decoder_output, decoder_hidden, attn_weights = self.forward_step(decoder_input,
                                                                             decoder_hidden,
                                                                             encoder_outputs)
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input
                

        decoder_output = torch.cat(decoder_outputs, dim=1)
        decoder_output = F.log_softmax(decoder_output, dim=-1)
        attentions = torch.cat(attentions, dim=1)
        
        return decoder_output, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))
        # embedding_shape: (N, 1, hidden_size)
        src_length = encoder_outputs.shape[1]
        query = hidden[-1].unsqueeze(1)
        query = query.repeat(1, src_length, 1)
        # query_shape: (N, seq_length, hidden_size)
        # encoder_outputs_shape: (N, seq_length, hidden_size)
        context, attn_weights = self.attention(query, encoder_outputs, encoder_outputs)
        output, hidden = self.rnn(embedded, hidden)
        input_out = torch.cat((output,context), dim=2)
        output = self.out(input_out)
        return output, hidden, attn_weights

    # Initialize network weights and biases using Xavier initialization
    
    def initialize_weights(self):
        for m in self.modules():
           
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    # count the number of parameters
    def num_parameters(self):
        n_params = 0
        for name, param in self.named_parameters():
            n_params += param.numel()
        return n_params



class DecoderRNN(nn.Module):
    """Decoder Layer for SeqToSeq Model.
    """

    def __init__(self,
                 hidden_size: int,
                 output_size: int,
                 n_layers: int,
                 max_length: int,
                 embedding_size: int,
                 dropout_p: float = 0.1,
                 **kwargs):
        """Initialize the DecoderRNN layer.

        Parameters
        ----------
        hidden_size: int
            Number of features in the hidden state.
        output_size: int
            Number of expected features.
        max_length: int
            Maximum length of the sequence.
        dropout_p: float (default 0.1)
            The dropout probability to use during training.
        """
        super(DecoderRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p, batch_first=True)
        
        self.out = nn.Linear(hidden_size, output_size)
        self.MAX_LENGTH = max_length
        self.num_layers = n_layers
        
        # Initialize the Net
        self.initialize_weights()
    
    def forward(self, encoder_outputs, fingerprints, target_tensor = None):
        """
        Parameters
        ----------
        inputs: List[torch.Tensor]
            A list of tensor containg encoder_hidden and target_tensor.

        Returns
        -------
        decoder_outputs: torch.Tensor
            Predicted output sequences.
        decoder_hidden: torch.Tensor
            Hidden state of the decoder.

        """
        
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.zeros(batch_size,
                                    1,
                                    dtype=torch.long,
                                    device=fingerprints.device)
        
        # laten space -> Interpreter
        z = F.relu(self.connection(fingerprints))
        decoder_hidden = torch.stack(self.num_layers * [z])
        
        decoder_outputs = []

        for i in range(self.MAX_LENGTH):
            
            decoder_output, decoder_hidden = self.forward_step(decoder_input,
                                                               decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_output = torch.cat(decoder_outputs, dim=1)
        decoder_output = F.log_softmax(decoder_output, dim=-1)
        return decoder_output, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        embedded = self.embedding(input)
        # Embedding shape: (N, 1, hidden_size)
        output = F.relu(embedded)
        output, hidden = self.rnn(output,hidden)
        # outputs shape: (N, 1, hidden_size)
        output = self.out(output)
        return output, hidden
    
    def initialize_weights(self):
        for m in self.modules():
           
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    # count the number of parameters
    def num_parameters(self):
        n_params = 0
        for name, param in self.named_parameters():
            n_params += param.numel()
        return n_params