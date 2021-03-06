from models import *
from utils.pytorch_utils import *
from embedders.embed import *
from datasets.YelpDataset import YelpDataset
from datasets.vocab_gen import Tokenizer, ByteBPETokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import logging

## https://arxiv.org/pdf/1804.00247.pdf Warmup training for transformer

def createLogger():
    console_logging_format = "%(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()

    return logger

def train(logger):
    
    ## ------ Dataset Modules ------- ##

    # tokenizer = Tokenizer("global", "datasets/vocabulary.txt")
    tokenizer = ByteBPETokenizer("datasets/yelp_bpe/yelp-bpe-vocab.json", "datasets/yelp_bpe/yelp-bpe-merges.txt", max_length=250)

    # 999 for usage of CLS token. 1000 if otherwise. Make sure to set add_cls=True if using CLS token
    training_yelp = YelpDataset("datasets/yelp_training.jsonl", tokenizer=tokenizer, max_len=250, is_from_partition=True, add_cls=False)
    validation_yelp = YelpDataset("datasets/yelp_validation.jsonl", tokenizer=tokenizer, max_len=250, is_from_partition=True, add_cls=False)

    #embedder = Embedding(tokenizer)
    #embedder.load_embedding("embedders/embedding_refine.txt.refine") # embedder.load_embedding("embedders/embeddingsV1.txt")
    #embedding_matrix = torch.Tensor(embedder.embed(200))

    logger.info("loaded dataset modules...")

    ## ------ Experiment Modules ------- ##

    epochs = 10
    batch_size = 32

    patience=2
    delta = 0
    checkpoint_file = "torch_transformer_v5_weight_BPE"
    model_file = "model_plots/{}".format(checkpoint_file)

    logger.info("Expiriment name: {}".format(checkpoint_file))

    training_loader = torch.utils.data.DataLoader(training_yelp, batch_size=batch_size, num_workers=8, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(validation_yelp, batch_size=batch_size, num_workers=8)

    vocab_size = 25000 #len(tokenizer.word2Index) #CLS TOKEN

    model = TorchTransformer(vocab_size, model_dim=256, ff_dim=512, num_heads=4, num_layers=4, num_classes=5, max_len=250, dropout=0.3, cls_token=False).cuda() #TorchBiLSTM(embedding_matrix, hidden_size=128, dropout=0.2).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1) # torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer_scheduler = WarmupLearninngRate(optimizer, warmup_steps = 10000, init_lr=0.07) #torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.9) 
    
    review_weight = torch.Tensor(np.array([ 0.8, 1.4, 1.4, 1.2, 0.8 ]))
    cross_entropy_loss = nn.CrossEntropyLoss(weight=review_weight).cuda()

    early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True, checkpoint_file=checkpoint_file+".pt")

    logger.info("loaded experiment modules...")

    ## ------ Training and Evaluation ----- ##
    training_losses, training_accuracies = [], []
    valid_losses, valid_accuracies = [], []

    for epoch in range(epochs):
        print("Starting Epoch {}/{}".format(epoch+1, epochs))

        t_avg_loss, t_avg_acc = train_epoch(model, training_loader, optimizer, optimizer_scheduler, cross_entropy_loss, epoch + 1)
        v_avg_loss, v_avg_acc = val_epoch(model, validation_loader, cross_entropy_loss, epoch+1)

        print("Completed Epoch {} Stats:\n Train Loss: {}; Train Acc: {};".format(epoch+1, t_avg_loss, t_avg_acc*100))
        print("Val Loss: {}; Val Acc: {};".format(v_avg_loss, v_avg_acc*100))
        
        training_losses.append(t_avg_loss)
        training_accuracies.append(t_avg_acc)
        valid_losses.append(v_avg_loss)
        valid_accuracies.append(v_avg_acc)

        # optimizer_scheduler.step()
        early_stopping(v_avg_loss, model)
        if early_stopping.early_stop:
            print("EARLY STOPPING...")
            break

    plot_and_save(training_losses, valid_losses, "Model Losses", "Loss", model_file + "_loss")
    plot_and_save(training_accuracies, valid_accuracies, "Model Accuracies", "Acc", model_file + "_acc")

def train_epoch(model, train_loader, optimizer, scheduler, loss_fn, epoch):
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

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        curr_accuracy = compute_accuracy(predicted_logits, targets, idx).item()
        total_accuracy += curr_accuracy

        avg_loss = total_loss / (idx + 1)

        loader.set_description("TRAIN - Avg Loss: %.4f; Curr. Accuracy: %.6f; LR: %.4f" % (avg_loss, curr_accuracy*100, \
                                                                                                optimizer.param_groups[0]['lr']) )
    
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