python generate_feature.py --test_path ../features/conll14 \
--out_path ../features/conll14.bver.ele \
--bert_pretrain google/electra-base-discriminator \
--checkpoint ../checkpoints/electra_model/model.best.pt

python generate_feature.py --test_path ../features/fce \
--out_path ../features/fce.bver.ele \
--bert_pretrain google/electra-base-discriminator \
--checkpoint ../checkpoints/electra_model/model.best.pt

python generate_feature.py --test_path ../features/bea19 \
--out_path ../features/bea19.bver.ele \
--bert_pretrain google/electra-base-discriminator \
--checkpoint ../checkpoints/electra_model/model.best.pt

python generate_feature.py --test_path ../features/jfleg \
--out_path ../features/jfleg.bver.ele \
--bert_pretrain google/electra-base-discriminator \
--checkpoint ../checkpoints/electra_model/model.best.pt

python generate_feature.py --test_path ../features/dev \
--out_path ../features/dev.bver.ele \
--bert_pretrain google/electra-base-discriminator \
--checkpoint ../checkpoints/electra_model/model.best.pt