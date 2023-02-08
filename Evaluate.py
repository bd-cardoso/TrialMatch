import os
import json
from ranx import Qrels, Run, evaluate  # https://amenra.github.io/ranx/
import pytrec_eval  # https://github.com/cvangysel/pytrec_eval
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import sys

if __name__ == '__main__':
    train_or_test = "train"
    # train_or_test = "test"
    run_name = "regex_train"
    # output_dir = f"/home/bd.cardoso/rework/BioLinkBERT/runs/2Classes-original-criteria/BioLinkBERT-large_SIGIR-CT-2A_epoch5"
    output_dir = sys.argv[1]
    run_name = sys.argv[2]
    run_file = f"{output_dir}/{run_name}.json"
    run_file = f"{output_dir}/{run_name}"
    with open(f"{run_file}", "r") as file:
        run_dict = json.load(file)
        run = Run(run_dict)

    if train_or_test == "train":
        f = "test"
    else:
        f = "train"

    with open(f"/home/bd.cardoso/my_datasets/trec_ct_2021/train_test_split/{f}_50.json", "r") as file:
        topics_test = json.load(file)
    with open(f"/home/bd.cardoso/my_datasets/trec_ct_2021/qrels/qrels_3_scale.json", "r") as file:
        qrels_3_dict = json.load(file)
        for topic_id in topics_test:
            del qrels_3_dict[str(topic_id)]
        qrels_3_scale = Qrels(qrels_3_dict)
    with open(f"/home/bd.cardoso/my_datasets/trec_ct_2021/qrels/qrels_2_scale_v1.json", "r") as file:
        qrels_2_scale_v1_dict = json.load(file)
        for topic_id in topics_test:
            del qrels_2_scale_v1_dict[str(topic_id)]
        qrels_2_scale_v1 = Qrels(qrels_2_scale_v1_dict)

    ndcg = evaluate(qrels_3_scale, run, "ndcg@10")

    results = evaluate(qrels_2_scale_v1, run,
                       ["precision@10", "r-precision", "mrr", "recall@10", "recall@100", "recall@500", "recall@1000",
                        "recall"])

    results.update({"ndcg@10": ndcg})

    for metric in results:
        results[metric] = round(results[metric], 4)
    pprint(results)

    with open(f"{output_dir}/{run_name}_metrics.json", "w") as file:
        json.dump(results, file, indent=4)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_2_scale_v1_dict, {"iprec_at_recall_0.00",
                                                                       "iprec_at_recall_0.10",
                                                                       "iprec_at_recall_0.20",
                                                                       "iprec_at_recall_0.30",
                                                                       "iprec_at_recall_0.40",
                                                                       "iprec_at_recall_0.50",
                                                                       "iprec_at_recall_0.60",
                                                                       "iprec_at_recall_0.70",
                                                                       "iprec_at_recall_0.80",
                                                                       "iprec_at_recall_0.90",
                                                                       "iprec_at_recall_1.00"})
    iprec_at_recall = evaluator.evaluate(run_dict)
    iprec_at_recall["mean"] = {}

    for query in iprec_at_recall:
        if query == "mean":
            continue
        for metric in iprec_at_recall[query]:
            if metric not in iprec_at_recall["mean"]:
                iprec_at_recall["mean"][metric] = []
            iprec_at_recall["mean"][metric].append(iprec_at_recall[query][metric])
    for metric in iprec_at_recall["mean"]:
        iprec_at_recall["mean"][metric] = np.mean(iprec_at_recall["mean"][metric])

    x, y = [], []
    for k, v in iprec_at_recall["mean"].items():
        x.append(k.split("_")[-1])
        y.append(v)

    # Precision-Recall Curve
    plt.style.use('seaborn')
    plt.plot(x, y, marker='o', linestyle='dashed')
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.savefig(f"{output_dir}/{run_name}_Precision-Recall.png", dpi=300, bbox_inches="tight")
