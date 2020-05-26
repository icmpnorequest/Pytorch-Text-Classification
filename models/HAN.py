# coding=utf-8
"""
@author: Yantong Lai
@date: 05/26/2020
@descrption: Reimplementation of hierachical attention network
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

import torchtext
from torchtext.data import get_tokenizer
from torchtext import data, datasets

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import spacy
spacy.load("en_core_web_sm")

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Dataset path
dataset_path = "../aclImdb"

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


####################################
#         Hyper-parameters         #
####################################
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 3
EMBEDDING_DIM = 300
WORD_RNN_SIZE = 100
OUTPUT_DIM = 2
DROPOUT = 0.5
WORD_ATTENTION_SIZE = 150
WORD_RNN_LAYERS = 2
BIDIRECTIONAL = True


####################################
#          Preparing Data          #
####################################
# 1. data.Field()
TEXT = data.Field(tokenize='spacy', include_lengths=True, batch_first=True)
LABEL = data.LabelField()

# 2. data.TabularDataset
train_data, valid_data, test_data = data.TabularDataset.splits(path=dataset_path,
                                                               train="train_cpu.tsv",
                                                               validation="valid_cpu.tsv",
                                                               test="test_cpu.tsv",
                                                               fields=[('label', LABEL), ('text', TEXT)],
                                                               format="tsv")
# train_data, test_data = datasets.IMDB.splits(TEXT, LABELS)
print("Number of train_data = {}".format(len(train_data)))
print("Number of valid_data = {}".format(len(valid_data)))
print("Number of test_data = {}".format(len(test_data)))

# 3. data.BucketIterator
train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                               batch_size=BATCH_SIZE,
                                                               device=device,
                                                               sort_key=lambda x: len(x.text))
# 4. Build vocab
TEXT.build_vocab(train_data)
VOCAB_SIZE = len(TEXT.vocab)
print("VOCAB_SIZE: {}".format(VOCAB_SIZE))
# , vectors="glove.6B.100d"
LABEL.build_vocab(train_data)
print("vars(train_data[0]) = {}\n".format(vars(train_data[0])))


####################################
#          Build the Model         #
####################################
class WordAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, word_rnn_size, word_rnn_layers, word_att_size, dropout):
        super(WordAttention, self).__init__()

        # 1. Embed
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # 2. Bidirectional word-level RNN
        self.word_rnn = nn.GRU(embedding_dim, word_rnn_size, num_layers=word_rnn_layers, bidirectional=True,
                               dropout=dropout, batch_first=True)

        # 3. Word-level attention network, MLP
        self.word_attention = nn.Linear(in_features=2 * word_rnn_size, out_features=word_att_size)

        # 4. Word context vector, to take dot-product with
        # y = W * word_context_vector + 0
        self.word_context_vector = nn.Linear(in_features=word_att_size, out_features=1, bias=False)

        # We could also do this with:
        # self.word_context_vector = nn.Parameter(torch.FloatTensor(1, word_att_size))
        # self.word_context_vector.data.uniform_(-0.1, 0.1)
        # And then take the dot-product

        # 5. Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_len):

        # 1. Get input size
        # print("1. Get text.size(): {}".format(text.size()))
        # text = [batch size, sent len]
        # text_len = [batch size]

        # 2. Embed
        embeddings = self.embeddings(text)
        embeddings = self.dropout(embeddings)
        print("2. Embed, embeddings.size(): {}".format(embeddings.size()))
        # embeddings = [batch size, sent len, embed dim]

        # 3. Pack words
        packed_words = pack_padded_sequence(input=embeddings, lengths=text_len, batch_first=True, enforce_sorted=False)
        print("3. packed_words.size(): {}".format(packed_words.data.size()))
        # packed_words.data = [n_words, embed dim]

        # 4. Bidirectional rnn
        packed_words, _ = self.word_rnn(packed_words)
        print("4. Bi-GRU, packed_words.size(): {}".format(packed_words.data.size()))
        # packed_words.data = [n_words, 2 * word_rnn_size]

        # 5. Attention
        # 5.1 Get Attention
        att_w = self.word_attention(packed_words.data)
        print("5.1 Attention, att_w.size(): {}".format(att_w.size()))
        # att_w = [n_words, word_att_size]
        # 5.2 tanh
        att_w = torch.tanh(att_w)
        print("5.2 Attention, att_w.size(): {}".format(att_w.size()))

        # 6. Get word context vector
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_w = self.word_context_vector(att_w).squeeze(1)
        print("6. Context vector, att_w.size(): {}".format(att_w.data.size()))
        # att_w = [n_words]

        # 7. Manually calculate word_alphas
        # 7.1 Get max value, scalar
        max_value = att_w.max()
        # 7.2 Calculate exp(att_w - max_value)
        att_w = torch.exp(att_w - max_value)
        # 7.3 Re-arrange as sentences by re-padding with 0s (WORDS -> SENTENCES)
        att_w, _ = pad_packed_sequence(PackedSequence(data=att_w,
                                                      batch_sizes=packed_words.batch_sizes,
                                                      sorted_indices=packed_words.sorted_indices,
                                                      unsorted_indices=packed_words.unsorted_indices),
                                       batch_first=True)
        print("7. pad_packed_sequence, att_w.size(): {}".format(att_w.data.size()))
        # att_w = [batch size, sent len]
        words_alphas = att_w / torch.sum(att_w, dim=1, keepdim=True)
        print("7. word_alphas.size(): {}".format(words_alphas.data.size()))
        # word_alphas = [batch size, sent len]

        # 8. Get sentences
        # 8.1 Similarly re-arrange word-level RNN outputs as sentences by re-padding with 0s (WORDS -> SENTENCES)
        sentences, _ = pad_packed_sequence(packed_words, batch_first=True)
        print("8.1 pad_packed_sequence, sentences.size(): {}".format(sentences.data.size()))
        # sentences = [batch size, sent len, 2 * word_rnn_size]
        # 8.2 Get dot-product of word_alphas and sentences
        sentences = sentences * (words_alphas.unsqueeze(2))
        print("8.2 sentences.size(): {}".format(sentences.data.size()))
        # sentences = [batch size, sent len, 2 * word_rnn_size]
        # 8.3 Sum
        sentences = sentences.sum(dim=1)
        print("8.3 sum, sentences.size(): {}".format(sentences.data.size()))
        # sentences = [batch size, 2 * word_rnn_size]

        return sentences, words_alphas


class SentenceAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, word_rnn_size, sentence_rnn_size, word_rnn_layers, sentence_rnn_layers,
                 word_att_size, sentence_att_size, dropout):
        """
        :param vocab_size: number of words in the vocabulary of model
        :param embedding_dim: size of word embeddings
        :param word_rnn_size: size of bidirectional word-level RNN
        :param sentence_rnn_size: size of bidirectional sentence-level RNN
        :param word_rnn_layers: number of layers in word-level RNN
        :param sentence_rnn_layers: number of layers in sentence-level RNN
        :param word_att_size: size of word-level attention layer
        :param sentence_att_size: size of sentence-level attention layer
        :param dropout: dropout
        """
        super(SentenceAttention, self).__init__()

        # 1. Word-level attention module
        self.word_attention = WordAttention(vocab_size=vocab_size, embedding_dim=embedding_dim,
                                            word_rnn_size=word_rnn_size, word_rnn_layers=word_rnn_layers,
                                            word_att_size=word_att_size, dropout=dropout)

        # 2. Bidirectional sentence-level RNN
        self.sentence_rnn = nn.GRU(2 * word_rnn_size, sentence_rnn_size, num_layers=sentence_rnn_layers,
                                   bidirectional=True, dropout=dropout, batch_first=True)

        # 3. Sentence-level attention network
        self.sentence_attention = nn.Linear(2 * sentence_rnn_size, sentence_att_size)

        # 4. Sentence context vector to take dot-product
        self.sentence_context_vector = nn.Linear(sentence_att_size, 1, bias=False)
        # You could also do this with:
        # self.sentence_context_vector = nn.Parameter(torch.FloatTensor(1, sentence_att_size))
        # self.sentence_context_vector.data.uniform_(-0.1, 0.1)
        # And then take the dot-product

        # 5. Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, documents, sentences_per_document, words_per_sentence):
        """
        :param documents: encoded document-level data, a tensor of dimensions (n_documents, sent_pad_len, word_pad_len)
        :param sentences_per_document: document lengths, a tensor of dimensions (n_documents)
        :param words_per_sentence: sentence lengths, a tensor of dimensions (n_documents, sent_pad_len)
        :return: document embeddings, attention weights of words, attention weights of sentences
        """
        # 1. Re-arrange as sentences by removing sentence pads (DOCUMENTS -> SENTENCES)
        packed_sentences = pack_padded_sequence(documents, lengths=sentences_per_document, batch_first=True,
                                                enforce_sorted=False)
        # packed_sentences = [n_sentences, word_pad_len]

        # 2. Re-arrange sentence lengths in the same way (DOCUMENTS -> SENTENCES)
        packed_words_per_sentence = pack_padded_sequence(words_per_sentence, lengths=sentences_per_document,
                                                         batch_first=True, enforce_sorted=False)
        # packed_words_per_sentence = [n_sentences]

        # 3. Get word-level attention sentences and words_alphas
        sentences, words_alphas = self.word_attention(packed_sentences.data, packed_words_per_sentence.data)
        # sentences = [batch size, 2 * word_rnn_size], words_alphas = [batch size, max(sent len)]
        sentences = self.dropout(sentences)

        # 4. Apply the sentence-level RNN over the sentence embeddings
        packed_sentences, _ = self.sentence_rnn(PackedSequence(data=sentences, batch_sizes=packed_sentences.batch_sizes,
                                                               sorted_indices=packed_sentences.sorted_indices,
                                                               unsorted_indices=packed_sentences.unsorted_indices))
        # packed_sentences = [n_sentences, 2 * sentence_rnn_size]

        # 5. Attention
        att_s = self.sentence_attention(packed_sentences.data)
        # att_s = [n_sentences, att_size]
        att_s = torch.tanh(att_s)
        att_s = self.sentence_context_vector(att_s).squeeze(1)
        # att_s = [n_sentences]

        # 6. Calculate sentence alphas
        # 6.1 Get max value of att_s, scalar
        max_value = att_s.max()
        # 6.2 Calculate exponent
        att_s = torch.exp(att_s - max_value)
        # att_s = [n_sentences]
        # 6.3 Re-arrange as documents by re-padding with 0s (SENTENCES -> DOCUMENTS)
        att_s, _ = pad_packed_sequence(PackedSequence(data=att_s, batch_sizes=packed_sentences.batch_sizes,
                                                      sorted_indices=packed_sentences.sorted_indices,
                                                      unsorted_indices=packed_sentences.unsorted_indices),
                                       batch_first=True)
        # att_s = [n_documents, max(sentences_per_document)]
        # 6.4 Calculate sentence alphas
        sentence_alphas = att_s / torch.sum(att_s, dim=1, keepdim=True)
        # sentence_alphas = [n_documents, max(sentences_per_document)]

        # 7. Similarly re-arrange sentence-level RNN outputs as documents by re-padding with 0s (SENTENCES -> DOCUMENTS)
        documents, _ = pad_packed_sequence(packed_sentences, batch_first=True)
        # documents = [n_documents, max(sentences_per_document), 2 * sentence_rnn_size]

        documents = documents * (sentence_alphas.unsqueeze(2))
        documents = documents.sum(dim=1)
        # documents = [n_documents, 2 * sentence_rnn_size]

        # Also re-arrange word_alphas (SENTENCES -> DOCUMENTS)
        word_alphas, _ = pad_packed_sequence(PackedSequence(data=words_alphas,
                                                            batch_sizes=packed_sentences.batch_sizes,
                                                            sorted_indices=packed_sentences.sorted_indices,
                                                            unsorted_indices=packed_sentences.unsorted_indices),
                                             batch_first=True)
        # word_alphas = [n_documents, max(sentences_per_document), max(words_per_sentence)]

        return documents, word_alphas, sentence_alphas


'''
wordAttention = WordAttention(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, word_rnn_size=WORD_RNN_SIZE,
                              output_dim=OUTPUT_DIM, word_rnn_layers=WORD_RNN_LAYERS, word_att_size=WORD_ATTENTION_SIZE,
                              dropout=DROPOUT)
wordAttention = wordAttention.to(device)


####################################
#          Train the Model         #
####################################
# criterion = nn.BCEWithLogitsLoss().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(wordAttention.parameters(), lr=LEARNING_RATE)

########## Train ##########
total_step = len(train_iter)
print("total_step = ", total_step)
for epoch in range(NUM_EPOCHS):
    wordAttention.train()
    total_loss = []
    train_total_correct = 0

    for i, batch in enumerate(train_iter):

        text, text_lengths = batch.text
        y = batch.label

        # Forward pass
        y_pred = wordAttention(text, text_lengths).squeeze(1)

        loss = criterion(y_pred, y)

        pred = torch.argmax(y_pred.data, dim=1)
        train_total_correct += (pred == y).sum().item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

        if (i + 1) % 2 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, loss.item()))

    print("total_loss = {}".format(total_loss))
    print("total_accuracy = {:.4f}%".format(100 * train_total_correct / len(train_data)))

'''




