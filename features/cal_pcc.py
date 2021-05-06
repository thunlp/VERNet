import math
def pearson_correlation(pred, ref):
    """ Computes Pearson correlation """
    from scipy.stats import pearsonr
    pc = pearsonr(pred, ref)
    return pc[0]  # return correlation value and ignore p,value



def test_file(prepath, goldpath):
    predict = list()
    gold = list()
    total_pcc = 0
    counter = 0
    with open(prepath) as fpre, open(goldpath) as fgold:
        lines = zip(fpre, fgold)
        for line in lines:
            predict.append(float(line[0].strip()))
            gold.append(float(line[1].strip()))
    step = len(predict) / 5
    for i in range(int(step)):
        pcc = pearson_correlation(predict[i*5: (i+1)*5], gold[i*5: (i+1)*5])
        if not math.isnan(pcc):
            total_pcc += pcc
            counter += 1
    total_pcc = round(total_pcc / counter, 4)
    print (total_pcc)

print ("-----------------------------")
'''test_file("conll14.bscore", "conll14.1.fscore")
test_file("conll14.rr", "conll14.1.fscore")
test_file("conll14.rc", "conll14.1.fscore")
test_file("conll14.cr", "conll14.1.fscore")
test_file("conll14.cc", "conll14.1.fscore")

test_file("conll14.blm", "conll14.1.fscore")
test_file("conll14.bgqe", "conll14.1.fscore")
test_file("conll14.bds", "conll14.1.fscore")


test_file("conll14.bqe", "conll14.1.fscore")
test_file("conll14.bdt", "conll14.1.fscore")
test_file("conll14.bdj", "conll14.1.fscore")
test_file("conll14.bver", "conll14.1.fscore")
test_file("conll14.bver.ele", "conll14.1.fscore")
test_file("conll14.bver", "conll14.0.fscore")
test_file("conll14.bver", "conll14.1.fscore")
test_file("fce.bver", "fce.fscore")
test_file("jfleg.bver", "jfleg.fscore")'''

test_file("conll14.bver", "conll14.0.fscore")
test_file("conll14.bver", "conll14.1.fscore")
test_file("fce.bver", "fce.fscore")
test_file("jfleg.bver", "jfleg.fscore")
print ("-----------------------------")
test_file("conll14.bver.ele", "conll14.0.fscore")
test_file("conll14.bver.ele", "conll14.1.fscore")
test_file("fce.bver.ele", "fce.fscore")
test_file("jfleg.bver.ele", "jfleg.fscore")
print ("-----------------------------")
test_file("conll14.bver.ablation", "conll14.0.fscore")
test_file("conll14.bver.ablation", "conll14.1.fscore")
test_file("fce.bver.ablation", "fce.fscore")
test_file("jfleg.bver.ablation", "jfleg.fscore")
print ("-----------------------------")

