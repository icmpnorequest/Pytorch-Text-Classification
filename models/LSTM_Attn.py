# _*_ coding: utf-8 _*_
"""
@author: Yantong Lai
@date: 04/13/2020
@descrption: Text classification with Attention mechanism in LSTM
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

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
BATCH_SIZE = 128
NUM_EPOCHS = 10
EMBEDDING_DIM = 300
HIDDEN_DIM = 100
OUTPUT_DIM = 2
DROPOUT = 0.5


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
# , vectors="glove.6B.100d"
LABEL.build_vocab(train_data)
print("vars(train_data[0]) = {}\n".format(vars(train_data[0])))


####################################
#          Build the Model         #
####################################
class AttentionModel(torch.nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(AttentionModel, self).__init__()
        self.hidden_dim = hidden_dim

        # 1. Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # self.word_embeddings.weights = nn.Parameter(weights, requires_grad=False)

        # 2. LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # 3. Linear layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def attention_net(self, lstm_output, final_state):

        hidden = final_state.squeeze(0)
        # print("Attention, hidden.size() = {}".format(hidden.size()))
        # hidden = [batch size, hidden dim]

        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        # print("attn_weights.size() = {}".format(attn_weights.size()))
        # attn_weights = [batch size, sent length]

        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state

    def forward(self, text, text_length):

        # 0. Text format
        # print("0. text.size() = {}".format(text.size()))
        # text.size() = [batch size, text length]

        # 1. Embed
        embedded = self.embedding(text)
        # print("1.1 embedded.size() = {}".format(embedded.size()))
        # 1. embedded.size() = torch.Size([1369, 128, 300])
        # [batch size, text length, embedding dim]
        embedded = embedded.permute(1, 0, 2)
        # print("1.2 embedded.size() = {}".format(embedded.size()))
        # [text length, batch size, embedding dim]

        # 2. Initial hidden and cell states
        # Set initial hidden and cell states
        h0 = torch.zeros(1, text.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(1, text.size(0), self.hidden_dim).to(device)
        # h0.size() = [1, batch size, hidden dim]

        # 3. LSTM
        lstm_output, (hidden, cell) = self.lstm(embedded, (h0, c0), )
        # print("3. lstm_output.size() = {}".format(lstm_output.size()))
        # print("3. hidden.size() = {}".format(hidden.size()))
        # hidden.size() = torch.Size([1, 128, 100])
        # hidden.size() = [1, batch size, hidden dim]

        lstm_output = lstm_output.permute(1, 0, 2)
        # packed_output.size() = [batch size, text length, hidden size]

        # 4. Attention
        attn_output = self.attention_net(lstm_output, hidden)
        # print("4. attn_out.size() = {}".format(attn_output.size()))

        # 5. Output
        logits = self.fc(attn_output)
        # print("5. logits.size() = {}".format(logits.size()))
        return logits

# Create an instance
INPUT_DIM = len(TEXT.vocab)
model = AttentionModel(vocab_size=INPUT_DIM, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                       output_dim=OUTPUT_DIM)
model = model.to(device)


####################################
#          Train the Model         #
####################################
# criterion = nn.BCEWithLogitsLoss().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

model = model.to(device)

########## Train ##########
total_step = len(train_iter)
print("total_step = ", total_step)
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = []
    train_total_correct = 0

    for i, batch in enumerate(train_iter):

        text, text_lengths = batch.text
        y = batch.label

        # Forward pass
        y_pred = model(text, text_lengths).squeeze(1)

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

    # Valid
    model.eval()
    valid_total_correct = 0
    with torch.no_grad():

        for i, batch in enumerate(valid_iter):
            text, text_lengths = batch.text
            y = batch.label

            # Forward pass
            y_pred = model(text, text_lengths).squeeze(1)

            loss = criterion(y_pred, y)

            pred = torch.argmax(y_pred.data, dim=1)
            valid_total_correct += (pred == y).sum().item()

        print("valid acc = {:.4f}%\n".format(100 * valid_total_correct / len(valid_data)))


####################################
#            Evaluation            #
####################################
model.eval()
test_correct = 0
test_loss = 0.0

for i, batch in enumerate(test_iter):
    text, text_lengths = batch.text
    y = batch.label

    # Forward pass
    # y_pred = model(text).squeeze(1).float()
    y_pred = model(text, text_lengths).squeeze(1)

    loss = criterion(y_pred, y)

    pred = torch.argmax(y_pred.data, dim=1)
    test_correct += (pred == y).sum().item()

    total_loss += loss.item()

    if (i + 1) % 2 == 0:
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, loss.item()))

avg_loss = test_loss / len(test_data)
print("Test Avg. Loss: {:.4f}, Accuracy: {:.4f}%"
      .format(avg_loss, 100 * test_correct / len(test_data)))