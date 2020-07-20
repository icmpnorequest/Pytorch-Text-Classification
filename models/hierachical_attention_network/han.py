# coding=utf-8
"""
@author: Yantong Lai
@date: 07/20/2020
@description: The implementation of hierachical attention network model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WordEncoder(nn.Module):
    """
    This class aims to encode words.
    input: word
    output: word encoding
    """
    def __init__(self, vocab_size, embed_dim, word_hidden_size):
        super(WordEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=vocab_size - 1)
        self.word_gru = nn.GRU(input_size=embed_dim, hidden_size=word_hidden_size, bidirectional=True)

    def forward(self, input):
        # print("WordEncoder. 0. input.size(): {}".format(input.size()))
        # input = [1, 64]

        word_embeddings = self.embedding(input)
        # word_embeddings = [1, 64, 2 * 50]
        # print("WordEncoder. 1. word_embeddings.size(): {}".format(word_embeddings.size()))

        f_output, h_output = self.word_gru(word_embeddings)
        # f_output = [1, 64, 2 * 50]
        # h_output = [2, 64, 50]. 2 represents bidirectional
        # print("WordEncoder. 2. f_output.size(): {}".format(f_output.size()))
        # print("WordEncoder. 2. h_output.size(): {}".format(h_output.size()))
        return f_output, h_output


class Attention(nn.Module):
    """
    This class aims to implement Attenion layer for words and sentences.
    input: words/sentences
    output: weighted sum of words/sentences
    """
    def __init__(self, input_size):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.fc = nn.Linear(self.input_size, self.input_size)
        self.context_vector = nn.Parameter(torch.randn(self.input_size))

    def forward(self, input):
        # print("Attention. 0. input.size(): {}".format(input.size()))
        # [64, 30, 100]

        # 1. Calculate u_{it}
        output = torch.tanh(self.fc(input))
        # output = [64, 30, 100]
        # print("Attention. 1. output.size(): {}".format(output.size()))

        # 2. Calculate alpha
        # 1) output X context_vector
        output = torch.matmul(output, self.context_vector)
        # output = [64, 30]
        # 2) Softmax
        output = F.softmax(output, dim=1)
        # output = [64, 30]

        # 3. Weighted sum
        output = output.permute(1, 0)
        # output = [30, 64]
        input = input.permute(1, 0, 2)
        # input = [30, 64, 100]

        batch_size = input.size()[1]
        weighted_sum = torch.zeros(batch_size, self.input_size)
        # Word. weighted_sum = [64, 100]
        # Sent. weighted_sum = [64, 160]

        for alpha, h in zip(output, input):
            # alpha = [64]
            # h = [64, 100]
            alpha = alpha.unsqueeze(1).expand_as(h)
            weighted_sum += alpha * h
        return weighted_sum


class SentenceEncoder(nn.Module):
    """
    This class aims to encode sentence.
    input: sentence
    output: sentence encoding
    """
    def __init__(self, input_size, sent_hidden_size):
        super(SentenceEncoder, self).__init__()
        self.sent_gru = nn.GRU(input_size, sent_hidden_size, bidirectional=True)

    def forward(self, input):
        f_output, h_output = self.sent_gru(input)
        return f_output, h_output


class HierachicalAttentionNetwork(nn.Module):
    """
    This class aims to create hierachical attention network model.
    input: <torch.Tensor> [batch size, sentences number, words number]
    output: <torch.Tensor> [batch size, num classes]
    """
    def __init__(self, vocab_size, embed_dim, word_hidden_size, sent_hidden_size, num_classes):
        super(HierachicalAttentionNetwork, self).__init__()

        # 1. Word encoder
        self.word_encoder = WordEncoder(vocab_size=vocab_size, embed_dim=embed_dim, word_hidden_size=word_hidden_size)

        # 2. Word attention
        self.word_attention = Attention(input_size=word_hidden_size * 2)    # bidirectional

        # 3. Sentence encoder
        self.sent_encoder = SentenceEncoder(input_size=word_hidden_size * 2, sent_hidden_size=sent_hidden_size)

        # 4. Sentence attention
        self.sent_attention = Attention(input_size=sent_hidden_size * 2)

        # 5. Output
        self.fc = nn.Linear(in_features=sent_hidden_size * 2, out_features=num_classes)

    def forward(self, input):
        # 1. Format input
        # input = [64, 2, 30]
        print("1. input.size(): {}".format(input.size()))
        input = input.permute(1, 2, 0)
        # input = [2, 30, 64]

        # 2. Get sentence from input
        sent_encoder_outputs = []
        for sentence in input:
            # sentence = [30, 64]

            word_encoder_outputs = []
            # 3. Get words of a sentence
            for word in sentence:
                # (1) Format word
                word = word.unsqueeze(0)
                # word = [1, 64]

                # (2) Encode word
                word_encoding, word_hidden_state = self.word_encoder(word)
                # word_encoding = [1, 64, 100]
                word_encoder_outputs.append(word_encoding)

            # (3) Concat <list> word_encoder_outputs to <torch.tensor>
            word_attn_inputs = torch.cat(word_encoder_outputs, dim=0)
            # word_attn_inputs = [30, 64, 100]

            # (4) Word attention
            word_attn_inputs = word_attn_inputs.permute(1, 0, 2)
            # word_attn_inputs = [64, 30, 100]
            word_attn_outputs = self.word_attention(word_attn_inputs)
            # word_attn_outputs = [64, 100]

            # (5) Encode sentence
            word_attn_outputs = word_attn_outputs.unsqueeze(0)
            # word_attn_outputs = [1, 64, 100]
            sent_encoding, sent_hidden_state = self.sent_encoder(word_attn_outputs)
            # sent_encoding = [1, 64, 160]
            sent_encoder_outputs.append(sent_encoding)

        # (6) Concat <list> sent_encoder_outputs to <torch.tensor>
        sent_attn_inputs = torch.cat(sent_encoder_outputs, dim=0)
        # sent_attn_inputs = [2, 64, 160]

        # (7) Sentence attention
        sent_attn_inputs = sent_attn_inputs.permute(1, 0, 2)
        # sent_attn_inputs = [64, 2, 160]
        sent_attn_outputs = self.sent_attention(sent_attn_inputs)
        # sent_attn_outputs = [64, 160]

        # 4. Output
        # (1) Fully connected layer
        output = self.fc(sent_attn_outputs)

        # (2) Log softmax
        output = F.log_softmax(output, dim=1)
        return output

