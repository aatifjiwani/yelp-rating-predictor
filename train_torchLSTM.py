from models.pytorch_lstm import TorchBiLSTM
from embedders.embed import *
from datasets.YelpDataset import YelpDataset
from datasets.vocab_gen import Tokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

import logging

def createLogger():
    console_logging_format = "%(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()

    return logger

def train(logger):
    tokenizer = Tokenizer("global", "datasets/vocabulary.txt")

    training_yelp = YelpDataset("datasets/yelp_training.jsonl", tokenizer=tokenizer, max_len=850, is_from_partition=True)
    validation_yelp = YelpDataset("datasets/yelp_validation.jsonl", tokenizer=tokenizer, max_len=850, is_from_partition=True)

    embedder = Embedding(tokenizer)
    # embedding_matrix = embedder.embedWithModel("embedders/embedded_version2.bin", 200)
    # embedding_matrix = torch.Tensor(embedding_matrix)
    embedder.load_embedding("embedders/embeddingsV1.txt")
    embedding_matrix = torch.Tensor(embedder.embed(200))

    logger.info("loaded dataset modules...")

    epochs = 10
    batch_size = 64

    training_loader = torch.utils.data.DataLoader(training_yelp, batch_size=128, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(validation_yelp, batch_size=128, num_workers=4)


    model = TorchBiLSTM(embedding_matrix, hidden_size=128, dropout=0.2).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2, momentum=0.9)
    cross_entropy_loss = F.cross_entropy

    logger.info("loaded experiment modules...")

    for epoch in range(epochs):
        print("Starting Epoch {}/{}".format(epoch+1, epochs))

        t_avg_loss, t_avg_acc = train_epoch(model, training_loader, optimizer, cross_entropy_loss, epoch + 1)
        v_avg_loss, v_avg_acc = val_epoch(mode, validation_loader, cross_entropy_loss, epoch+1)
        
        print("Completed Epoch {} Stats:\n Train Loss: {}; Train Acc: {};".format(epoch+1, t_avg_loss, t_avg_acc*100))
        print("Val Loss: {}; Val Acc: {};".format(v_avg_loss, v_avg_acc*100))

def train_epoch(model, train_loader, optimizer, loss_fn, epoch):
    total_loss = 0
    total_accuracy = 0
    loader = tqdm(train_loader)

    model.train()
    for idx, inputs in enumerate(loader):
        optimizer.zero_grad()

        reviews, targets = inputs["input"], inputs["label"]
        reviews, targets = reviews.cuda(), targets.cuda()

        predicted_logits = model(reviews)
        loss = loss_fn(predicted_logits, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        curr_accuracy = compute_accuracy(predicted_logits, targets, idx).item()
        total_accuracy += curr_accuracy

        avg_loss = total_loss / (idx + 1)
        loader.set_description("TRAIN - Avg Loss: %.4f; Curr. Accuracy: %.6f;" % (avg_loss, curr_accuracy*100) )
    
    return avg_loss, total_accuracy / (idx + 1)

def val_epoch(model, val_loader, loss_fn, epoch):
    total_loss = 0
    total_accuracy = 0
    loader = tqdm(val_loader)

    model.eval()
    for idx, inputs in enumerate(loader):
        reviews, targets = inputs["input"], inputs["label"]
        reviews, targets = reviews.cuda(), targets.cuda()

        predicted_logits = model(reviews)
        loss = loss_fn(predicted_logits, targets)

        total_loss += loss.item()
        curr_accuracy = compute_accuracy(predicted_logits, targets, idx).item()
        total_accuracy += curr_accuracy

        avg_loss = total_loss / (idx + 1)
        loader.set_description("VALIDATION - Avg Loss: %.4f; Curr. Accuracy: %.6f;" % (avg_loss, curr_accuracy*100) )
    
    return avg_loss, total_accuracy / (idx + 1)


def compute_accuracy(logits, target, idx):

    pred = logits.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum()

    accuracy = correct / float(target.shape[0])

    return accuracy

if __name__ == "__main__":
    logger = createLogger()
    train(logger)