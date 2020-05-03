import json, sys
from test_torchLSTM import *

from models.pytorch_lstm import TorchBiLSTM
from utils.pytorch_utils import *
from embedders.embed import *
from datasets.YelpDataset import YelpDataset
from datasets.vocab_gen import Tokenizer

from tqdm import tqdm

checkpoint_file = "torch_bilstm_v3_lr1e3.pt"
tokenizer = Tokenizer("global", "datasets/vocabulary.txt")

embedder = Embedding(tokenizer)
embedder.load_embedding("embedders/embeddingsV1.txt")
embedding_matrix = torch.Tensor(embedder.embed(200))

model = TorchBiLSTM(embedding_matrix, hidden_size=128, dropout=0.2).cuda()
model.load_state_dict(torch.load("model_checkpoints/{}".format(checkpoint_file)))
model.eval()

def eval(text):
	# This is where you call your model to get the number of stars output
	star = test_with_text(text, model, YelpDataset, tokenizer)
	return star

if len(sys.argv) > 1:
	validation_file = sys.argv[1]
	print("Generating output file")
	with open("output.jsonl", "w") as fw:
		with open(validation_file, "r") as fr:
			for line in tqdm(fr):
				review = json.loads(line)
				fw.write(json.dumps({"review_id": review['review_id'], "predicted_stars": eval(review['text'])})+"\n")
	print("Output prediction file written")
else:
	print("No validation file given")


# text = "I went to this campus for 1 semester. I was in Business - Information Systems.\n\nThe campus is okay. The food choices are bismal.\n\nThe building is laid with the cafeteria on the bottom level, and then classes on the 2nd, 3rd, and 4th with each faculty basically having their own floor.\n\nTHe campus is pretty enough, but have fun getting the elevator around class start times...you're better to just stair it. \n\n\nIt's Seneca College after all."
# print(eval(text))