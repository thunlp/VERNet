import argparse
import numpy as np
data_features = list()
scores = list()
parser = argparse.ArgumentParser()
parser.add_argument('--feature_path', nargs='+')
parser.add_argument("--score_path")
parser.add_argument("--out_path")
parser.add_argument('--nbest', type=int, default=5)
args = parser.parse_args()

for path in args.feature_path:
    features = list()
    with open(path) as fin:
        for line in fin:
            features.append(float(line.strip()))
    if len(data_features) == 0:
        for _ in range(len(features)):
            data_features.append([])
    elif len(data_features) != len(features):
        raise ("Feature instance wrong!")
    for step, feature in enumerate(features):
        if ".blm" in path:
            feature = np.exp(feature)
        data_features[step].append(feature)

rel_score = np.zeros([len(data_features)//args.nbest, args.nbest])
if args.score_path:
    with open(args.score_path) as fin:
        for line in fin:
            scores.append(float(line.strip()))
    assert len(scores) == len(data_features)
    scores = np.array(scores)
    scores = scores.reshape(-1, args.nbest)
    rel_score = np.zeros(scores.shape)
    for i in range(len(scores)):
        max_score = 0.0
        for j in range(len(scores[i])):
            if scores[i][j] > max_score:
                max_score = scores[i][j]
        for j in range(len(scores[i])):
            if scores[i][j] == max_score and scores[i][j] != 0:
                rel_score[i][j] = 1


data_features = np.array(data_features)
data_features = data_features.reshape(len(rel_score), args.nbest, len(args.feature_path))
with open(args.out_path, "w") as fout:
    for i in range(len(data_features)):
        for j in range(len(data_features[i])):
            string_list = list()
            string_list.append(str(rel_score[i][j]))
            string_list.append("qid:"+str(i + 1))
            for k in range(len(data_features[i][j])):
                string_list.append(str(k+1) + ":" + str(data_features[i][j][k]))
            fout.write(" ".join(string_list) + "\n")





















