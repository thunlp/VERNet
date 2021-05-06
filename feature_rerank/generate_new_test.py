def process_file(inpath, scorepath, outpath, number, max_num):
    data_list = list()
    src_path = inpath + ".src"
    hyp_path = inpath + ".hyp"
    score_path = scorepath
    with open(src_path) as src, open(hyp_path) as hyp, open(score_path) as score:
        lines = list(zip(src, hyp, score))
        for step, line in enumerate(lines):
            counter = int(step / max_num)
            if counter >= len(data_list):
                data_list.append([])
            data_list[counter].append([line[1].strip(), float(line[2].strip().split()[-1])])
    with open(outpath + str(number), "w") as fout:
        for step in range(len(data_list)):
            values = data_list[step][:number + 1]
            values = sorted(values, key=lambda x:x[1], reverse=True)
            fout.write(values[0][0]+ "\n")
if __name__ == "__main__":
    for i in range(5):
        process_file("../features/bea19", "./bea19.score.txt", "./bea19.rerank", i, 5)
        process_file("../features/fce", "./fce.score.txt", "./fce.rerank", i, 5)
        process_file("../features/conll14", "./conll14.score.txt", "./conll14.rerank", i, 5)
        process_file("../features/jfleg", "./jfleg.score.txt", "./jfleg.rerank", i, 5)
