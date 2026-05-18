# dd2417_project


## Commands
`python -m training.train_tokenizer --texts data/hamlet.txt --output model/tokenizer.json --vocab-size 5000`
`python -m training.train_transformer --texts data/hamlet.txt --tokenizer model/tokenizer.json --checkpoint model/checkpoint.pt --epochs 2 --batch-size 8 --block-size 256`
`python -m training.train_word_counts --texts data/hamlet.txt --output model/word_counts.pkl`
`python src/demo_predict.py`

