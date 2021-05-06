java -jar RankLib-2.1-patched.jar -load model_electra.txt -rank fce.feature -score fce.score.txt
java -jar RankLib-2.1-patched.jar -load model_electra.txt -rank bea19.feature -score bea19.score.txt
java -jar RankLib-2.1-patched.jar -load model_electra.txt -rank conll14.feature -score conll14.score.txt
java -jar RankLib-2.1-patched.jar -load model_electra.txt -rank jfleg.feature -score jfleg.score.txt
python generate_new_test.py

python2 ../m2scorer/scripts/m2scorer.py ./conll14.rerank1  ../data/test.m2
python2 ../m2scorer/scripts/m2scorer.py ./conll14.rerank2  ../data/test.m2
python2 ../m2scorer/scripts/m2scorer.py ./conll14.rerank3  ../data/test.m2
python2 ../m2scorer/scripts/m2scorer.py ./conll14.rerank4  ../data/test.m2

python ../jfleg/eval/gleu.py -r ../jfleg/test/test.ref[0-3] --hyp ./jfleg.rerank1 --src ../jfleg/test/test.src
python ../jfleg/eval/gleu.py -r ../jfleg/test/test.ref[0-3] --hyp ./jfleg.rerank2 --src ../jfleg/test/test.src
python ../jfleg/eval/gleu.py -r ../jfleg/test/test.ref[0-3] --hyp ./jfleg.rerank3 --src ../jfleg/test/test.src
python ../jfleg/eval/gleu.py -r ../jfleg/test/test.ref[0-3] --hyp ./jfleg.rerank4 --src ../jfleg/test/test.src

errant_parallel -orig ../data/fce.src -cor fce.rerank1 -out fce.1.m2
errant_parallel -orig ../data/fce.src -cor fce.rerank2 -out fce.2.m2
errant_parallel -orig ../data/fce.src -cor fce.rerank3 -out fce.3.m2
errant_parallel -orig ../data/fce.src -cor fce.rerank4 -out fce.4.m2

errant_parallel -orig ../data/conll14.src -cor conll14.rerank1 -out conll14.1.m2
errant_parallel -orig ../data/conll14.src -cor conll14.rerank2 -out conll14.2.m2
errant_parallel -orig ../data/conll14.src -cor conll14.rerank3 -out conll14.3.m2
errant_parallel -orig ../data/conll14.src -cor conll14.rerank4 -out conll14.4.m2

errant_compare -hyp conll14.1.m2 -ref ../data/conll14.m2
errant_compare -hyp conll14.2.m2 -ref ../data/conll14.m2
errant_compare -hyp conll14.3.m2 -ref ../data/conll14.m2
errant_compare -hyp conll14.4.m2 -ref ../data/conll14.m2

errant_compare -hyp fce.1.m2 -ref ../data/fce.m2
errant_compare -hyp fce.2.m2 -ref ../data/fce.m2
errant_compare -hyp fce.3.m2 -ref ../data/fce.m2
errant_compare -hyp fce.4.m2 -ref ../data/fce.m2
