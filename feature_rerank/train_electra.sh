python get_features.py --feature_path ../features/dev.bscore ../features/dev.bver.ele --score_path ../features/dev.fscore --out_path ./train.ele.feature
python get_features.py --feature_path ../features/conll14.bscore ../features/conll14.bver.ele --score_path ../features/conll14.fscore --out_path ./conll14.feature
python get_features.py --feature_path ../features/fce.bscore ../features/fce.bver.ele --score_path ../features/fce.fscore --out_path ./fce.feature
python get_features.py --feature_path ../features/bea19.bscore ../features/bea19.bver.ele --out_path ./bea19.feature
python get_features.py --feature_path ../features/jfleg.bscore ../features/jfleg.bver.ele --out_path ./jfleg.feature

java -jar RankLib-2.1-patched.jar -train ./train.ele.feature  -ranker 4 -metric2t MAP -metric2T MAP -save model_electra.txt
