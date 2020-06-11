import torch
import torch.nn as nn

class ELMo(nn.Module):
    def __init__(self, opts, vocab_sizes):
        super(ELMo, self).__init__()
        self.opts = opts
        self.word_vocab_size, self.char_size = vocab_sizes
        self.char_embeddings = nn.Embedding(self.char_size, self.opts.char_embedding_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.char_embeddings.weight)
        self.dropout = nn.Dropout(opts.dropout)

        # Build char-BiLSTM
        self.char_lstm = nn.LSTM(opts.char_embedding_dim, opts.char_embedding_dim // 2,
                                 bidirectional=True, batch_first=True)

        # Build 2 layer Bi-directional LSTM
        self.forward_lstm1 = nn.LSTM(opts.projection_dim, opts.hidden_dim,
                            num_layers=opts.num_layers, batch_first=True)

        self.forward_lstm2 = nn.LSTM(opts.projection_dim, opts.hidden_dim,
                            num_layers=opts.num_layers, batch_first=True)

        self.backward_lstm1 = nn.LSTM(opts.projection_dim, opts.hidden_dim,
                            num_layers=opts.num_layers, batch_first=True)

        self.backward_lstm2 = nn.LSTM(opts.projection_dim, opts.hidden_dim,
                            num_layers=opts.num_layers, batch_first=True)

        # projection matrix
        self.forward_projection1 = nn.Linear(opts.hidden_dim, opts.projection_dim, bias=True)
        self.backward_projection1 = nn.Linear(opts.hidden_dim, opts.projection_dim, bias=True)
        self.forward_projection2 = nn.Linear(opts.hidden_dim, opts.projection_dim, bias=True)
        self.backward_projection2 = nn.Linear(opts.hidden_dim, opts.projection_dim, bias=True)

        # Embedding layer
        self.char_fc_layer = nn.Linear(self.opts.char_fc_dim, self.opts.projection_dim)

        # Maps the output of the LSTM into tag space.
        self.softmax_fc_layer = nn.Linear(self.opts.projection_dim * 2, self.word_vocab_size)

    def forward(self, word_ids, char_ids):
        self.char_lstm.flatten_parameters()
        self.forward_lstm1.flatten_parameters()
        self.forward_lstm2.flatten_parameters()
        self.backward_lstm1.flatten_parameters()
        self.backward_lstm2.flatten_parameters()

        char_embedding = self.dropout(self.char_embeddings(char_ids).view(-1, char_ids.size()[2], self.opts.char_embedding_dim))
        char_representation, (h_n, c_n) = self.char_lstm(char_embedding)
        char_representation = char_representation.view(-1, char_ids.size()[1], char_ids.size()[2], self.opts.char_embedding_dim)
        char_representation = char_representation.reshape(char_representation.size()[0], char_ids.size()[1], -1)
        char_representation = self.char_fc_layer(char_representation)

        forward_input = char_representation[:,:-1,:]
        forward_output1, (h_n, c_n) = self.forward_lstm1(forward_input)
        proj_forward_output1 = self.dropout(self.forward_projection1(forward_output1) + char_representation[:,:-1,:]) # residual connection
        forward_output2, (h_n, c_n) = self.forward_lstm2(proj_forward_output1)
        proj_forward_output2 = self.dropout(self.forward_projection2(forward_output2) + proj_forward_output1) # residual connection

        backward_input = torch.flip(char_representation, [1])[:,:-1,:]
        backward_output1, (h_n, c_n) = self.backward_lstm1(backward_input)
        proj_backward_output1 = self.dropout(self.backward_projection1(backward_output1) + torch.flip(char_representation, [1])[:,:-1,:]) # residual connection
        backward_output2, (h_n, c_n) = self.backward_lstm2(proj_backward_output1)
        proj_backward_output2 = self.dropout(self.backward_projection2(backward_output2) + proj_backward_output1) # residual connection


        output = torch.cat([proj_forward_output2, proj_backward_output2], dim=-1)

        softmax = self.softmax_fc_layer(output).view(-1, self.word_vocab_size)

        return softmax
