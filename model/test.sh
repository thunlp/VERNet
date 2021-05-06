python test.py --test_path ../data/fce \
--bert_pretrain bert-base-cased \
--checkpoint ../checkpoints/bert_model/model.best.pt

python test.py --test_path ../data/conll14.0 \
--bert_pretrain bert-base-cased \
--checkpoint ../checkpoints/bert_model/model.best.pt

python test.py --test_path ../data/conll14.1 \
--bert_pretrain bert-base-cased \
--checkpoint ../checkpoints/bert_model/model.best.pt

python test_src.py --test_path ../data/fce \
--bert_pretrain bert-base-cased \
--checkpoint ../checkpoints/bert_model/model.best.pt

python test_src.py --test_path ../data/conll14.0 \
--bert_pretrain bert-base-cased \
--checkpoint ../checkpoints/bert_model/model.best.pt

python test_src.py --test_path ../data/conll14.1 \
--bert_pretrain bert-base-cased \
--checkpoint ../checkpoints/bert_model/model.best.pt