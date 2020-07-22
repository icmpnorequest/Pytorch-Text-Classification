# coding=utf-8
"""
@author: Yantong Lai
@date: 07/21/2020
@description: HAN experiment with AgNews dataset
"""

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os

from models.hierachical_attention_network.utils import EmbedDataHelper
from models.hierachical_attention_network.dataset import AgNewsDataset
from models.hierachical_attention_network.han import HierachicalAttentionNetwork


# ------------------------------#
#       1. Definitions          #
# ------------------------------#
# processed ndarray
processed_train_docs_file = "../../agnews/processed_train_docs.npy"
processed_val_docs_file = "../../agnews/processed_val_docs.npy"
processed_test_docs_file = "../../agnews/processed_test_docs.npy"
processed_train_labels_file = "../../agnews/processed_train_labels.npy"
processed_val_labels_file = "../../agnews/processed_val_labels.npy"
processed_test_labels_file = "../../agnews/processed_test_labels.npy"

# embed file
embed_file_path = "../../glove/glove.6B.100d.txt"

# device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get vocab from glove
embedDataHelper = EmbedDataHelper(embed_file=embed_file_path)
keys, values = embedDataHelper.load_vocab()
keys_dict = embedDataHelper.generate_keys_dict(keys)

# hyper-parameters
EMBED_DIM = 200
WORD_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 80
NUM_EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MOMENTUM = 0.9
VOCAB_SIZE = embedDataHelper.get_vocab_size(keys)
NUM_CLASSES = 4

# trained model path
if not os.path.exists("../../trained_model/"):
    os.mkdir("../../trained_model/")
model_path = "../../trained_model/AgNews_HAN.pt"


# ------------------------------#
#    2. Data Initialization     #
# ------------------------------#
# (1) Create Dataset objects
train_dataset = AgNewsDataset(docs_file=processed_train_docs_file, labels_file=processed_train_labels_file)
val_dataset = AgNewsDataset(docs_file=processed_val_docs_file, labels_file=processed_val_labels_file)
test_dataset = AgNewsDataset(docs_file=processed_test_docs_file, labels_file=processed_test_labels_file)

# (2) Create Dataloader objects
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
print("Data initialized already.")


# ------------------------------#
#   3. Model Initialization     #
# ------------------------------#
# (1) Create HierachicalAttentionNetwork instance
model = HierachicalAttentionNetwork(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, word_hidden_size=WORD_HIDDEN_SIZE,
                                    sent_hidden_size=SENT_HIDDEN_SIZE, num_classes=NUM_CLASSES)

# (2) Define criterion and optimizer
criterion = torch.nn.NLLLoss().to(DEVICE)
optimizer = torch.optim.SGD(
    (p for p in model.parameters() if p.requires_grad),
    lr=LEARNING_RATE,
    momentum=MOMENTUM)


# ------------------------------#
#      4. Train the Model       #
# ------------------------------#
# (1) Train
total_step = len(train_dataloader)
print("total_step = {}\n".format(total_step))
# total_step = 1757

print("(1) Now, starts training.")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = []
    train_total_correct = 0

    for i, (label, text) in enumerate(train_dataloader):

        # Forward pass
        y_pred = model(text).squeeze(1)

        # Calculate loss
        loss = criterion(y_pred, label)

        # Get predicted label
        pred = torch.argmax(y_pred.data, dim=1)
        train_total_correct += (pred == label).sum().item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Collect loss
        total_loss.append(loss.item())

        # Print every five step
        if (i + 1) % 1 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, loss.item()))

    print("total_loss = {}".format(total_loss))
    print("total_accuracy = {:.4f}%".format(100 * train_total_correct / len(train_dataset)))

    # (2) Valid
    print("(2) Now, starts validation.")
    model.eval()
    valid_total_correct = 0
    with torch.no_grad():

        for i, (label, text) in enumerate(val_dataloader):

            # Forward pass
            y_pred = model(text).squeeze(1)

            # Calculate loss
            loss = criterion(y_pred, label)

            # Get prediction
            pred = torch.argmax(y_pred.data, dim=1)
            valid_total_correct += (pred == label).sum().item()

        print("valid acc = {:.4f}%\n".format(100 * valid_total_correct / len(val_dataset)))

# (3) Save trained model to specific path
torch.save(model.state_dict(), model_path)

