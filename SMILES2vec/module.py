''' This file includes modules needed for training, basically the neural nets '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class batch_generator():
    '''
    USAGE:  generator = batch_generator( batch_size = 50): 
            x_batch, y_batch = generator.next_batch(X_train, y_train)

    This function uses static method to count for next batch
    Note that caller is responsible to determine that how many rounds there should be in each epoch!
    For example:
        for epoch in range(total_epoch) :
            for small_round in range( len(y_train)/ self.batch_size )
                X_batch, y_batch = next_batch(X,y, self.batch_size)
                do something here
    '''
    def __init__(self,batch_size=100):
        self.batch_size = batch_size
        self.static_counter = 0
        self.new_round=False
        
    def next_batch(self, X, y):
        ''' should be same length for X and y '''
        if self.static_counter==None:
            self.static_counter = 0
        data_size = len(y)
        if ( self.static_counter+1 ) * self.batch_size >= data_size:
            self.static_counter = 0
            self.new_round = True
            return X[ data_size - self.batch_size: ], y[data_size - self.batch_size : ]
        else:
            self.static_counter += 1
            self.new_round = False
            start, end = self.batch_size * ( self.static_counter -1 ) , self.batch_size * self.static_counter
            return X[start: end], y[start: end]

class decoder_gru(nn.Module):
    ''' implement different decoder for debugging 
        this decoder use dense_vector and output_vector as next input of decoder RNN
    '''
    def __init__(self, feature_size, dense_size, device):
        super().__init__()
        self.feature_size = feature_size
        self.dense_size = dense_size
        self.device = device
        self.embedding_size = 50
        self.linear1 = nn.Sequential(
            nn.Linear(dense_size, dense_size),
            nn.SELU()    
        )
        self.GRU1 = nn.GRU(input_size=dense_size+feature_size, hidden_size=self.embedding_size)
        self.GRU2 = nn.GRU(input_size=self.embedding_size, hidden_size=self.embedding_size)
        self.linear2 = nn.Linear(self.embedding_size, feature_size)
    def forward(self, input_dense, seq, input_onehot, teacher_forcing,):
        dense_size = self.linear1(input_dense)
        hidden1 = torch.zeros(1,1,self.embedding_size).to(self.device)
        hidden2 = torch.zeros(1,1,self.embedding_size).to(self.device)
        output3 = torch.zeros(1,1,self.feature_size).to(self.device)
        next_input = torch.cat([input_dense, output3], dim=2)
        li = []
        for i in range(seq):
            output1, hidden1 = self.GRU1(next_input, hidden1)
            output2, hidden2 = self.GRU2(output1, hidden2)
            output3 = self.linear2(output2)
            li.append(output3)
            if teacher_forcing:
                next_input = torch.cat([input_dense, input_onehot[i].reshape(1,1,-1)], dim=2)
            else:
                # we use onehot input for next stage
                next_input = torch.cat([input_dense, F.softmax(output3, dim=2)], dim=2)
        return torch.cat(li)

class autoencoder_gru(nn.Module):
    def __init__(self, feature_size, dense_size, device):
        ''' ::input_size:: the size of input feature size
            ::dense_size:: the size of encoded dense vector feature size
        '''
        super().__init__()
        self.feature_size = feature_size
        self.dense_size = dense_size
        self.embedding_size = 10
        self.layer_size = 2
        self.device = device
        #self.E_hidden_size = 15 # size between GRU
        #self.D_hidden_size = 20
        
        # Encoder part
        self.embedding = nn.Embedding(feature_size, self.embedding_size)
        self.E_GRU = nn.GRU(input_size=self.embedding_size, hidden_size=dense_size, num_layers=self.layer_size)
        
        # Decoder part
        self.decoder = decoder_gru(feature_size, dense_size, device)
        #initialization
        
    def forward(self, input_mol, input_onehot=None, teacher_forcing=False):
        ''' produce final encoding '''
        self.seq = len(input_mol)
        output = self.encode(input_mol)
        output = output[0][-1].reshape(1,1,-1)
        # output: [1,1,dense_mol]
        #output = torch.cat([output for i in range(seq)]).reshape(seq,1,-1)
        output = self.decode(output, input_onehot, teacher_forcing )
        output = F.softmax(output, dim=2)
        return output
        
    def encode(self, input_mol):
        # input_mol: [seq,1]
        embedded = self.embedding(input_mol)
        # embedded: [seq,1,embedding_size]
        zero_hidden = torch.zeros(input_mol.size(1) * self.layer_size, 1, self.dense_size).to(self.device)
        output = self.E_GRU(embedded, zero_hidden)
        return output
        
    def decode(self, dense_mol, input_onehot, teacher_forcing):
        return self.decoder(dense_mol, self.seq, input_onehot, teacher_forcing)

class VAE_gru(nn.Module):
    def __init__(self, feature_size, dense_size, device):
        ''' ::input_size:: the size of input feature size
            ::dense_size:: the size of encoded dense vector feature size
        '''
        super().__init__()
        self.feature_size = feature_size
        self.dense_size = dense_size
        self.embedding_size = 12
        self.layer_size = 3
        self.device = device
        #self.E_hidden_size = 15 # size between GRU
        #self.D_hidden_size = 20
        
        # Encoder part
        self.embedding = nn.Embedding(feature_size, self.embedding_size)
        self.E_GRU_mu = nn.GRU(input_size=self.embedding_size, hidden_size=dense_size, num_layers=self.layer_size)
        self.E_GRU_logvar = nn.GRU(input_size=self.embedding_size, hidden_size=dense_size, num_layers=self.layer_size)
        
        # Decoder part
        self.decoder = decoder_gru(feature_size, dense_size, device)
        #initialization
        
    def forward(self, input_mol, input_onehot=None, teacher_forcing=False, variation=True):
        ''' produce final encoding '''
        self.seq = len(input_mol)
        mu, logvar = self.encode(input_mol)
        output = self.reparameterize(mu, logvar, variation).reshape(1,1,-1)
        # output: [1,1,dense_mol]
        #output = torch.cat([output for i in range(seq)]).reshape(seq,1,-1)
        output = self.decode(output, input_onehot, teacher_forcing )
        output = F.softmax(output, dim=2)
        return output
        
    def encode(self, input_mol):
        # input_mol: [seq,1]
        embedded = self.embedding(input_mol)
        # embedded: [seq,1,embedding_size]
        zero_hidden = torch.zeros(input_mol.size(1) * self.layer_size, 1, self.dense_size).to(self.device)
        mu = self.E_GRU_mu(embedded, zero_hidden)[0][-1]
        logvar = self.E_GRU_logvar(embedded, zero_hidden)[0][-1]
        return mu, logvar
        
    def decode(self, dense_mol, input_onehot, teacher_forcing):
        return self.decoder(dense_mol, self.seq, input_onehot, teacher_forcing)

    def reparameterize(self, mu, logvar, variation=True):
        ''' ::variation:: switch for turning sampling variation on/off  '''
        if variation:
            std = torch.exp(0.1 * logvar) # the constant controls the variation
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu.to(self.device)