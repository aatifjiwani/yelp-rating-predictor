# yelp-rating-predictor
A yelp rating predictor built using NLP models to predict the star rating of a written review.

Instructions
 - Install required packages from requirements.txt
 
 - Use test_submission.py to test the model on various datasets
    Datasets must be a .jsonl file and obey the following format
      {"review_id": "0", "text": "...", "stars": "1.0"}
      {"review_id": "1", "text": "...", "stars": "5.0"}
      
- Output predictions will be in output.jsonl 
  Please remember to rename before testing to another dataset otherwise predictions will be overwritten.
      
