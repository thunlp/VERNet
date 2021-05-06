# Rerank Beam Search Candidates with VERNet

* ``get_features.py``: Process ranking features to the Ranklib format.
* ``RankLib-2.1-patched.jar``: Learning-to-rank package.
* ``generate_new_test.py``: The script is used to get top-1 reranked results with the aggregated score.
* ``train.sh``: process BERT-VERNet features and train Coordinate Ascent models to reweight features.
* ``train_electra.sh``: Process ELECTRA-VERNet features and train Coordinate Ascent models to reweight features.
* ``test.sh``: Get aggregated score of the features from BERT-VERNet and basic GEC model and automatically evaluate GEC results.
* ``test_electra.sh``: Get aggregated score of the features from ELECTRA-VERNet and basic GEC model and automatically evaluate GEC results.
