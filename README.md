# Yelp Me Rate This: Using Deep Learning to Predict Yelp Review Ratings
A yelp rating predictor built using NLP models to predict the star rating of a written review.

## Instructions on running project/testing models
### 1. Clone the repository using 
```
git clone https://github.com/aatifjiwani/yelp-rating-predictor.git
```
### 2. Download the model checkpoints from the following link: 

https://drive.google.com/drive/folders/1hiW7-aJJoUsPAjYbuK8ymuz2YupcExYE?usp=sharing

Place all checkpoints within ```model_checkpoints/``` for ease of running the scripts without FileNotFound errors. 

The ```model_checks.zip``` file contains the model checkpoints from two BiLSTMs, 7 Transformers, and a RoBERTa model. If you've read the project report, the following models correspond to these checkpoints:<br>
- BiLSTM-Concat --> ```torch_bilstm_v1.pt``` (Requires Base Tokenizer)
- BiLSTM-Sum    --> ```torch_bilstm_v3_1e3.pt``` (Requires Base Tokenizer)
- Transformer-256 --> ```torch_transformer_v1.pt``` (Requires Base Tokenizer)
- Transformer-360-WCE-BPE --> ```torch_transformer_v4_weight.pt``` (Requires BPE Tokenizer)
- Transformer-5Layer-WCE-BPE --> ```torch_transformer_v7_weight.pt```(Requires BPE Tokenizer)
- RoBERTa --> ```checkpoint_3000/```(Tokenizer already included within model checkpoint) [2nd BEST MODEL]

The ```XLNet.zip``` file contains the model checkpoints for the XLNet. The zip file in its entirety is a smaller duplicate of the entire codebase but only utilize ```models/xlnet-checkpoint-30000/```. Move this directory into the same directory as the above model checkpoints. 

### 3. Install required packages from requirements.txt using 
``` 
pip install -r requirements.txt
```
**NOTE**: If you're using a Mac OS/Windows, ensure you have Python 3.6 as ```simpletransformers``` requires it. If you're using Linux, ensure you have Python 3.7. 

### 4. Testing/Running Models

**IMPORTANT**: All of non pre-trained Transformer models utilize ```YelpDataset``` in ```datasets/YelpDataset.py```. For initial usage, you must pass in a tokenizer object (see ```datasets/vocab_gen.py```).

An example of instantiating a Base Tokenizer and BPE Tokenizer is as follows:
```
base_tokenizer = Tokenizer("global", "datasets/vocabulary.txt")
bpe_tokenizer = ByteBPETokenizer("datasets/yelp_bpe/yelp-bpe-vocab.json", "datasets/yelp_bpe/yelp-bpe-merges.txt", max_length=250)
    
base_yelp = YelpDataset("datasets/yelp_challenge_3.jsonl", tokenizer=base_tokenizer, max_len=1000, is_from_partition=False)
bpe_yelp = YelpDataset("datasets/yelp_challenge_3.jsonl", tokenizer=bpe_tokenizer, max_len=250, is_from_partition=False)

base_loader = torch.utils.data.DataLoader(base_yelp, batch_size=32, num_workers=4, shuffle=False)    
bpe_loader = torch.utils.data.DataLoader(bpe_yelp, batch_size=32, num_workers=4, shuffle=False)   
```

For more information on how to load the LSTMs and Transformers, refer to ```test_submission.py``` and ```test_torchModel.py```. Please remember that the models ending in ```BPE``` require a Byte-Level BPE Tokenizer. Refer to ```test_torchModel.py``` on how to specify the tokenizer in the YelpDataset. 

For information on how to load RoBERTa or XLNet, please refer to ```test_submission_v2.py```. Loading these models are much easier than the custom PyTorch models. Also, note that it is much faster to test these pre-trained transformers when all the reviews are cleaned and in a 2D array. These models will automatically tokenize these sentences and pad if needed to the max length. 
 
Testing datasets MUST be in a ```.jsonl``` format and obey the following convention:
```
{"review_id": "0", "text": "...", "stars": "1.0"}
{"review_id": "1", "text": "...", "stars": "5.0"}
...
{"review_id": "XYZ", "text": "...", "stars": "3.0"}
```
To test an entire dataset all at once, use 
```
python3 test_submission<_v2>.py dataset_name.jsonl
```

Output predictions will be in ```output.jsonl```. Please remember to rename before testing to another dataset otherwise predictions will be overwritten. If you want to test a different model that is not defaulted in ```test_submission.py```, simply copy and paste the correct lines from ```test_torchModel.py```.
      
