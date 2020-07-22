# coding=utf-8
"""
@author: Yantong Lai
@date: 07/22/2020
"""

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from models.hierachical_attention_network.utils import EmbedDataHelper
from models.hierachical_attention_network.dataset import AgNewsDataset
from models.hierachical_attention_network.han import HierachicalAttentionNetwork


# ------------------------------#
#       1. Definitions          #
# ------------------------------#
# processed ndarray
processed_test_docs_file = "../../agnews/processed_test_docs.npy"
processed_test_labels_file = "../../agnews/processed_test_labels.npy"

# embed file
embed_file_path = "../../glove/glove.6B.100d.txt"

# device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# trained model path
model_path = "../../trained_model/AgNews_HAN.pt"

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


# ------------------------------#
#    2. Data Initialization     #
# ------------------------------#
# (1) Create Dataset objects
test_dataset = AgNewsDataset(docs_file=processed_test_docs_file, labels_file=processed_test_labels_file)

# (2) Create Dataloader objects
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
print("Data initialized already.")


# ------------------------------#
#         3. Load Model         #
# ------------------------------#
# (1) Model initialization
model = HierachicalAttentionNetwork(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, word_hidden_size=WORD_HIDDEN_SIZE,
                                    sent_hidden_size=SENT_HIDDEN_SIZE, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))

# (2) Loss function
criterion = torch.nn.NLLLoss().to(DEVICE)


# ------------------------------#
#        4. Evaluation          #
# ------------------------------#
model.eval()
test_correct = 0
test_loss = 0.0

with torch.no_grad():
    for i, (label, text) in enumerate(test_dataloader):
        label = label.to(DEVICE)
        text = text.to(DEVICE)

        # Forward pass
        y_pred = model(text).squeeze(1)

        # Calculate loss
        loss = criterion(y_pred, label)

        # Get prediction
        pred = torch.argmax(y_pred.data, dim=1)
        test_correct += (pred == label).sum().item()
        test_loss += loss.item()

avg_loss = test_loss / len(test_dataset)
print("Test Avg. Loss: {:.4f}, Accuracy: {:.4f}%"
      .format(avg_loss, 100 * test_correct / len(test_dataset)))

# Test Avg. Loss: 0.0085, Accuracy: 80.0921%

