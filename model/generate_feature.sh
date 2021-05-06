python generate_feature.py --test_path ../features/conll14 \
--out_path ../features/conll14.bver \
--bert_pretrain bert-base-cased \
--checkpoint ../checkpoints/bert_model/model.best.pt

python generate_feature.py --test_path ../features/fce \
--out_path ../features/fce.bver \
--bert_pretrain bert-base-cased \
--checkpoint ../checkpoints/bert_model/model.best.pt

python generate_feature.py --test_path ../features/bea19 \
--out_path ../features/bea19.bver \
--bert_pretrain bert-base-cased \
--checkpoint ../checkpoints/bert_model/model.best.pt

python generate_feature.py --test_path ../features/jfleg \
--out_path ../features/jfleg.bver \
--bert_pretrain bert-base-cased \
--checkpoint ../checkpoints/bert_model/model.best.pt

python generate_feature.py --test_path ../features/dev \
--out_path ../features/dev.bver \
--bert_pretrain bert-base-cased \
--checkpoint ../checkpoints/bert_model/model.best.pt