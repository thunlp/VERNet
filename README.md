#  Neural Verification Network (VERNet)
There are source codes for Neural Quality Estimation with Multiple Hypotheses for Grammatical Error Correction.

![model](https://github.com/thunlp/VERNet/blob/master/model.png)



## Requirement
* Python==3.6.2
* Pytorch==1.2.0
* transformers==3.2.0
* pip3 install errant
* Clone the ``jfleg`` project from here ``https://github.com/keisks/jfleg``


## Data and Checkpoint
* All these files can be downloaded and you should put them in the corresponding folders.
* All ``data`` can be found at [Ali Drive](https://thunlp.oss-cn-qingdao.aliyuncs.com/VERNet/data.zip).
* The ``checkpoints`` (BERT-VERNet and ELECTRA-VERNet) can be found at [Ali Drive](https://thunlp.oss-cn-qingdao.aliyuncs.com/VERNet/checkpoints.zip).
* All ``features`` used in reranking can be found at [Ali Drive](https://thunlp.oss-cn-qingdao.aliyuncs.com/VERNet/features.zip).

## Train VERNet
* VERNet inherits Hugginface Transformers, you can change codes for various pretrained language models.
* Go to the ``model`` folder and train models with BERT or ELECTRA as follow:
```
bash train.sh
```
```
bash train_electra.sh
```


## Test Token Level Quality Estimation Ability
* These experimental results are shown in Table 3 of our paper.
* The evaluations are the same as the evaluations of GED models.
* Go to the ``model`` folder and test BERT-VERNet model or ELECTRA-VERNet model as follow:
```
bash test.sh
```
```
bash test_electra.sh
```

## Rerank Beam Search Candidates
* First you should go to the ``model`` folder and generate features from BERT-VERNet model or ELECTRA-VERNet model. So run the following command:
```
bash generate_feature.sh
```
```
bash generate_feature_electra.sh
```

* (Optional Stage for Learning Feature Weight) Second, you can generate ranking features with the ``GEC model`` score and ``VERNet`` score, all these results are provided in the ``features`` folder. And then we train learning-to-rank models with Coordinate Ascent and get weights of ranking features. You should go to the ``feature_rerank`` folder and run the following command:
```
bash train.sh
```
```
bash train_electra.sh
```

* Finally, if you want to test the model, you can go to the ``feature_rerank`` folder and run the following command:
```
bash test.sh
```
```
bash test_electra.sh
```
 Using this command, you can rerank beam search candidates and automatically evaluate the model performance on three datasets CoNLL-2014, FCE and JFLEG. For BEA19 evaluation, you should submit the runs to their hidden test [website](https://competitions.codalab.org/competitions/20228).



## Results
The results are shown as follows.

|                        | CoNLL2014 (M2) |        |        | CoNLL2014 (ERRANT) |        |        | FCE    |        |        | BEA19  |        |        | JFLEG  |
|------------------------|----------------|--------|--------|--------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
|                        | P              | R      | F0.5   | P                  | R      | F0.5   | P      | R      | F0.5   | P      | R      | F0.5   | GLEU   |
| Basic GEC              | 68.59          | 44.87  | 62.03  | 64.26              | 43.59  | 58.69  | 55.11  | 41.61  | 51.75  | 66.20  | 61.40  | 65.20  | 61.00  |
| Basic GEC w. R2L       | 72.40          | 46.10  | 65.00  | -                  | -      | -      | -      | -      | -      | 74.70  | 56.70  | 70.20  | 61.40  |
| BERT-fuse (GED)        | 69.20          | 45.60  | 62.60  | -                  | -      | -      | -      | -      | -      | 67.10  | 60.10  | 65.60  | 61.30  |
| BERT-fuse (GED) w. R2L | 72.60          | 46.40  | 65.20  | -                  | -      | -      | -      | -      | -      | 72.30  | 61.40  | 69.80  | 62.00  |
| BERT-VERNet (Top2)     | 69.98          | 43.60  | 62.47  | 65.62              | 41.90  | 58.98  | 58.57  | 41.53  | 54.13  | 68.42  | 60.30  | 66.63  | 61.17  |
| BERT-VERNet (Top3)     | 70.49          | 43.16  | 62.50  | 65.92              | 41.22  | 58.86  | 59.20  | 41.53  | 54.55  | 69.03  | 60.20  | 67.06  | 61.20  |
| BERT-VERNet (Top4)     | 70.70          | 42.72  | 62.56  | 66.60              | 40.94  | 59.20  | 59.55  | 41.50  | 54.80  | 69.40  | 60.17  | 67.30  | 61.16  |
| BERT-VERNet (Top5)     | 70.60          | 42.50  | 62.36  | 66.41              | 40.74  | 58.98  | 59.60  | 41.48  | 54.80  | 69.39  | 60.12  | 67.32  | 61.10  |
| ELECTRA-VERNet (Top2)  | 71.21          | 44.20  | 63.47  | 66.95              | 42.90  | 60.22  | 58.31  | 41.97  | 54.09  | 69.27  | 61.22  | 67.50  | 61.60  |
| ELECTRA-VERNet (Top3)  | 71.80          | 44.13  | 63.80  | 67.50              | 42.38  | 60.30  | 59.02  | 41.99  | 54.59  | 70.64  | 61.78  | 68.67  | 61.80  |
| ELECTRA-VERNet (Top4)  | 71.85          | 43.81  | 63.69  | 67.48              | 42.19  | 60.25  | 59.65  | 42.12  | 55.07  | 70.90  | 62.00  | 68.90  | 62.00  |
| ELECTRA-VERNet (Top5)  | 71.58          | 43.57  | 63.43  | 67.15              | 42.10  | 60.01  | 59.90  | 42.10  | 55.20  | 70.79  | 61.74  | 68.77  | 62.07  |







## Citation
```
@inproceedings{liu2021vernet,
  title={Neural Quality Estimation with Multiple Hypotheses for Grammatical Error Correction},
  author={Liu, Zhenghao and Yi, Xiaoyuan and Sun, Maosong and Yang, Liner and Chua, Tat-Seng},
  booktitle={Proceedings of NAACL},
  year={2021}
}
```

## Contact
If you have questions, suggestions, and bug reports, please email:
```
liu-zh16@mails.tsinghua.edu.cn
```
