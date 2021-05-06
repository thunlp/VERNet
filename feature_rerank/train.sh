python get_features.py --feature_path ../features/dev.bscore ../features/dev.bver --score_path ../features/dev.fscore --out_path ./train.feature
python get_features.py --feature_path ../features/conll14.bscore ../features/conll14.bver --score_path ../features/conll14.fscore --out_path ./conll14.feature
python get_features.py --feature_path ../features/fce.bscore ../features/fce.bver --score_path ../features/fce.fscore --out_path ./fce.feature
python get_features.py --feature_path ../features/bea19.bscore ../features/bea19.bver --out_path ./bea19.feature
python get_features.py --feature_path ../features/jfleg.bscore ../features/jfleg.bver --out_path ./jfleg.feature

java -jar RankLib-2.1-patched.jar -train ./train.feature  -ranker 4 -metric2t MAP -metric2T MAP -save model_bert.txt
