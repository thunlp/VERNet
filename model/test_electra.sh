python test.py --test_path ../data/fce \
--checkpoint ../checkpoints/electra_model/model.best.pt \
--bert_pretrain google/electra-base-discriminator

python test.py --test_path ../data/conll14.0 \
--bert_pretrain google/electra-base-discriminator \
--checkpoint ../checkpoints/electra_model/model.best.pt

python test.py --test_path ../data/conll14.1 \
--bert_pretrain google/electra-base-discriminator \
--checkpoint ../checkpoints/electra_model/model.best.pt

python test_src.py --test_path ../data/fce \
--checkpoint ../checkpoints/electra_model/model.best.pt \
--bert_pretrain google/electra-base-discriminator

python test_src.py --test_path ../data/conll14.0 \
--bert_pretrain google/electra-base-discriminator \
--checkpoint ../checkpoints/electra_model/model.best.pt

python test_src.py --test_path ../data/conll14.1 \
--bert_pretrain google/electra-base-discriminator \
--checkpoint ../checkpoints/electra_model/model.best.pt
