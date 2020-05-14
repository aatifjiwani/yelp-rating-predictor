import json, sys
import tqdm
from models.pretrained_bert import PretrainedBert
from datasets.vocab_gen import *
import time

start = time.time()
b = PretrainedBert()
b.load('models/checkpoint-30000')

def eval(text):
        # This is where you call your model to get the number of stars output
        stars, _ = b.eval(text)
        return stars

if len(sys.argv) > 1:
        validation_file = sys.argv[1]
        print("Generating output file")
        with open("output.jsonl", "w") as fw:
                with open(validation_file, "r") as fr:
                        reviews = []
                        for line in tqdm(fr):
                                reviews.append(json.loads(line))
                        evals = eval([clean_sentence(x['text']) for x in reviews])
                        for i in range(len(evals)):
                                fw.write(json.dumps({"review_id": reviews[i]['review_id'], "predicted_stars": int(evals[i])+1})+"\n")
        print("Output prediction file written")
else:
        print("No validation file given")
print(time.time()-start)