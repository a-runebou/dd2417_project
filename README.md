# dd2417_project


## Commands
`python src/train_tokenizer.py --texts data/HP_book_1.txt data/hamlet.txt data/the_king_in_yellow.txt data/naval.txt --output data/tokenizer.json --vocab-size 15000`
`python src/train_transformer.py --texts data/HP_book_1.txt data/hamlet.txt data/the_king_in_yellow.txt data/naval.txt --tokenizer data/tokenizer.json --checkpoint data/checkpoint.pt --epochs 5 --batch-size 8 --block-size 256`
`python src/train_word_counts.py --texts data/HP_book_1.txt data/hamlet.txt data/the_king_in_yellow.txt data/naval.txt --output data/word_counts.pkl`
`python src/demo_predict.py `