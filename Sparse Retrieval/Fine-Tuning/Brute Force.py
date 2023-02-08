import json
import pandas as pd
import numpy as np
from ranx import Qrels, Run, evaluate
from multiprocessing import Pool, cpu_count

with open(f"/home/bd.cardoso/my_datasets/trec_ct_2021/train_test_split/test_50.json", "r") as file:
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

with open(f"/home/bd.cardoso/my_datasets/trec_ct_2021/topics/topics_expanded.json", "r") as file:
    topics_expanded = json.load(file)

with open(f"/home/bd.cardoso/my_datasets/trec_ct_2021/train_test_split/train_50.json", "r") as file:
    topics_train = json.load(file)


def evaluate_trec_ct(run):
    ndcg = evaluate(qrels_3_scale, run, "ndcg@10", threads=1)
    results = evaluate(qrels_2_scale_v1, run,
                       ["precision@10", "r-precision", "mrr", "recall@10", "recall@100", "recall@500",
                        "recall@1000"], threads=1)
    results.update({"ndcg@10": ndcg})
    for metric in results:
        results[metric] = round(results[metric], 4)
    return results


def retrieve(params):
    from pyserini.search.lucene import LuceneSearcher
    from pyserini.trectools import TrecRun
    from pyserini.fusion import reciprocal_rank_fusion

    print(f'Starting {int(params[6])}', flush=True)

    K = 1000
    searcher = LuceneSearcher("/home/bd.cardoso/pyserini/indexes/trec_ct_2021/all_free_text_fields")
    searcher.set_bm25(k1=float(params[1]), b=float(params[0]))
    searcher.set_rm3(fb_terms=int(params[2]), fb_docs=int(params[3]), original_query_weight=float(params[4]))

    runs = []
    for topic_id in topics_expanded:
        if int(topic_id) in topics_train:
            for query_id in topics_expanded[topic_id]:
                query = topics_expanded[topic_id][query_id]

                hits = searcher.search(query, k=K)

                rows = []
                for rank, hit in enumerate(hits, start=1):
                    rows.append((topic_id, 'Q0', hit.docid, rank, hit.score, f'{topic_id}_{query_id}'))
                runs.append(TrecRun.from_list(rows))

    run = reciprocal_rank_fusion(runs=runs, rrf_k=int(params[5]), depth=K, k=K)
    run = pd.DataFrame(data=run.to_numpy(), columns=['q_id', '_1', 'doc_id', '_2', 'score', '_3'])
    run = run.astype({'score': 'float'})
    run = Run.from_df(run)

    evaluation = evaluate_trec_ct(run)

    df = pd.DataFrame(
        data=[[float(params[1]), float(params[0]), int(params[2]), int(params[3]), float(params[4]), int(params[5]),
               evaluation['ndcg@10'], evaluation['precision@10'], evaluation['r-precision'], evaluation['mrr'],
               evaluation['recall@10'], evaluation['recall@100'], evaluation['recall@500'], evaluation['recall@1000']]],
        columns=['k1', 'b', 'fb_terms', 'fb_docs', 'original_query_weight', 'rrf_k', 'ndcg@10', "precision@10",
                 "r-precision", "mrr", "recall@10", "recall@100", "recall@500", "recall@1000"])

    df.to_csv(f'results/{int(params[6])}.csv', index=False)
    print(f'Done with {int(params[6])}', flush=True)


if __name__ == '__main__':
    setup = np.genfromtxt('setup.tsv', delimiter='\t')

    print(f'Number of cpus: {cpu_count()}', flush=True)

    p = Pool(100)
    with p:
        p.map(retrieve, setup)
