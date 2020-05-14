from models import *
from utils.pytorch_utils import *
from embedders.embed import *
from datasets.YelpDataset import YelpDataset
from datasets.vocab_gen import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import logging

import numpy as np

def createLogger():
    console_logging_format = "%(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()

    return logger

def test_with_text(text, model, DatasetClass, tokenizer):
    # checkpoint_file = "torch_bilstm_v3_lr1e3.pt"

    # tokenizer = Tokenizer("global", "datasets/vocabulary.txt")

    # embedder = Embedding(tokenizer)
    # embedder.load_embedding("embedders/embeddingsV1.txt")
    # embedding_matrix = torch.Tensor(embedder.embed(200))

    # model = TorchBiLSTM(embedding_matrix, hidden_size=128, dropout=0.2).cuda()
    # model.load_state_dict(torch.load("model_checkpoints/{}".format(checkpoint_file)))
    # model.eval()

    review = DatasetClass.getFromText(text, tokenizer)
    review = torch.Tensor(np.expand_dims(review, axis=0)).long().cuda()

    predicted_logits = model(review)

    stars = predicted_logits.argmax(dim=1)[0].item()

    return stars + 1 #MAKE SURE TO ADD 1


def test_with_answers(logger):
    
    ## ------ Dataset Modules ------- ##

    base_tokenizer = Tokenizer("global", "datasets/vocabulary.txt")
    bpe_tokenizer = ByteBPETokenizer("datasets/yelp_bpe/yelp-bpe-vocab.json", "datasets/yelp_bpe/yelp-bpe-merges.txt", max_length=250)
    base_yelp = YelpDataset("datasets/yelp_challenge_3.jsonl", tokenizer=base_tokenizer, max_len=1000, is_from_partition=False)
    bpe_yelp = YelpDataset("datasets/yelp_challenge_3.jsonl", tokenizer=bpe_tokenizer, max_len=250, is_from_partition=False)

    embedder = Embedding(base_tokenizer)
    embedder.load_embedding("embedders/embeddingsV1.txt")
    embedding_matrix = torch.Tensor(embedder.embed(200))

    logger.info("loaded dataset modules...")

    ## ------ Experiment Modules ------- ##

    batch_size = 1
    trans_checkpoint_file = "torch_transformer_v4_weight_BPE.pt"
    trans_model = TorchTransformer(25000, model_dim=360, ff_dim=512, num_heads=4, num_layers=4, num_classes=5, max_len=250, dropout=0.3, cls_token=False).cuda() 
    trans_model.load_state_dict(torch.load("model_checkpoints/{}".format(trans_checkpoint_file)))
    trans_model.eval()

    lstm_checkpoint_file = "torch_bilstm_v3_lr1e3.pt"
    lstm_model = TorchBiLSTM(embedding_matrix, hidden_size=128, dropout=0.2).cuda()
    lstm_model.load_state_dict(torch.load("model_checkpoints/{}".format(lstm_checkpoint_file)))
    lstm_model.eval()

    bert_model = PretrainedBert()
    bert_model.load('model_checkpoints/checkpoint-30000') 

    ## ------ Testing ----- ##

    logger.info("loaded experiment modules...")

    groundtruth_stars = np.array([int(base_yelp.reviews[x]['label']) - 1 for x in range(len(base_yelp))])

    bert_reviews = [clean_sentence(base_yelp.reviews[x]['input']) for x in range(len(base_yelp))]
    bert_eval = eval_bert(bert_model, bert_reviews)

    base_loader = torch.utils.data.DataLoader(base_yelp, batch_size=len(base_yelp), num_workers=4, shuffle=False)    
    loader = tqdm(base_loader)

    for idx, inputs in enumerate(loader):
        reviews, targets = inputs["input"], inputs["label"]
        reviews, targets = reviews.cuda(), targets.cuda()

        predicted_logits = lstm_model(reviews).argmax(dim=1, keepdim=True)
        lstm_eval = np.squeeze(predicted_logits.cpu().numpy())

    bpe_loader = torch.utils.data.DataLoader(bpe_yelp, batch_size=32, num_workers=4, shuffle=False)    
    b_loader = tqdm(bpe_loader)
    trans_eval = np.array([])

    for idx, inputs in enumerate(b_loader):
        reviews, targets = inputs["input"], inputs["label"]
        reviews, targets = reviews.cuda(), targets.cuda()

        predicted_logits = trans_model(reviews).argmax(dim=1, keepdim=True)
        trans_eval = np.append(trans_eval, predicted_logits.cpu().numpy())

    # print(bert_eval.shape)
    # print(lstm_eval.shape)
    # print(trans_eval.shape)

    # print(bert_eval[:25])
    # print(lstm_eval[:25])
    # print(trans_eval[:25])

    total_examples = len(groundtruth_stars)
    mean_abs_error = 0
    mean_accuracy = 0
    for idx in tqdm(range(len(bert_eval))):
        c = Counter()
        bert_star, lstm_star, trans_star = bert_eval[idx], lstm_eval[idx], trans_eval[idx]
        target_star = groundtruth_stars[idx]
        c[bert_star] += 1
        c[lstm_star] += 1
        c[trans_star] += 1

        if len(c) <= 2:
            pred_star = max(c)
        else:
            pred_star = int(np.ceil( np.average( list(c) ) ))


        mean_abs_error += abs(pred_star - target_star)
        mean_accuracy += 1 if pred_star == target_star else 0

        # targets = targets.view_as(predicted_logits)
        
        # abs_error = (torch.abs(predicted_logits - targets)).sum()
        # accuracy = predicted_logits.eq(targets).sum()

        # mean_abs_error += abs_error.item()
        # mean_accuracy += accuracy.item()
        # total_examples += reviews.shape[0]

    
    print("MEAN ABSOLUTE ERROR: {}".format((mean_abs_error / total_examples)))
    print("MEAN ACCURACY: {}".format((mean_accuracy / total_examples)))

def eval_torch(model, review):
    pass

def eval_bert(model, text):
    stars, _ = model.eval(text)
    return stars

if __name__ == "__main__":
    logger = createLogger()
    test_with_answers(logger)

    # text = "I went to this campus for 1 semester. I was in Business - Information Systems.\n\nThe campus is okay. The food choices are bismal.\n\nThe building is laid with the cafeteria on the bottom level, and then classes on the 2nd, 3rd, and 4th with each faculty basically having their own floor.\n\nTHe campus is pretty enough, but have fun getting the elevator around class start times...you're better to just stair it. \n\n\nIt's Seneca College after all."
    # test_with_text(text)