from models.pytorch_lstm import TorchBiLSTM
from utils.pytorch_utils import *
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

def test_with_answers(logger):
    
    ## ------ Dataset Modules ------- ##

    tokenizer = Tokenizer("global", "datasets/vocabulary.txt")
    testing_yelp = YelpDataset("datasets/yelp_challenge_7.jsonl", tokenizer=tokenizer, max_len=1000, is_from_partition=False)

    embedder = Embedding(tokenizer)
    embedder.load_embedding("embedders/embeddingsV1.txt")
    embedding_matrix = torch.Tensor(embedder.embed(200))

    logger.info("loaded dataset modules...")

    ## ------ Experiment Modules ------- ##

    batch_size = 2
    checkpoint_file = "torch_bilstm_v3_lr1e3.pt"

    test_loader = torch.utils.data.DataLoader(testing_yelp, batch_size=batch_size, num_workers=2, shuffle=False)

    model = TorchBiLSTM(embedding_matrix, hidden_size=128, dropout=0.2).cuda()
    model.load_state_dict(torch.load("model_checkpoints/{}".format(checkpoint_file)))
    model.eval()

    ## ------ Testing ----- ##

    logger.info("loaded experiment modules...")

    loader = tqdm(test_loader)


    total_examples = 0
    mean_abs_error = 0
    mean_accuracy = 0
    for idx, inputs in enumerate(loader):
        reviews, targets = inputs["input"], inputs["label"]
        reviews, targets = reviews.cuda(), targets.cuda()

        predicted_logits = model(reviews).argmax(dim=1, keepdim=True)
        targets = targets.view_as(predicted_logits)
        
        abs_error = (predicted_logits - targets).sum()
        accuracy = predicted_logits.eq(targets).sum()

        mean_abs_error += abs(abs_error.item())
        mean_accuracy += accuracy.item()
        total_examples += reviews.shape[0]

    
    print("MEAN ABSOLUTE ERROR: {}".format((mean_abs_error / total_examples)))
    print("MEAN ACCURACY: {}".format((mean_accuracy / total_examples)))

if __name__ == "__main__":
    logger = createLogger()
    test_with_answers(logger)