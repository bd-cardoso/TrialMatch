import json
from pyserini.search.lucene import LuceneSearcher
from pyserini.trectools import TrecRun
from pyserini.fusion import reciprocal_rank_fusion
from ranx import Qrels, Run, evaluate
import pandas as pd

K = 1000

index = "/home/bd.cardoso/pyserini/indexes/trec_ct_2021/all_free_text_fields"

with open("/home/bd.cardoso/my_datasets/trec_ct_2021/topics/topics_expanded.json", "r") as r:
    topics_expanded = json.load(r)

with open("/home/bd.cardoso/my_datasets/trec_ct_2021/qrels/qrels_3_scale.json", "r") as r:
    qrels_3_scale = json.load(r)

with open("/home/bd.cardoso/my_datasets/trec_ct_2021/qrels/qrels_2_scale_v1.json", "r") as r:
    qrels_2_scale_v1 = json.load(r)

run_dict = {}
searcher = LuceneSearcher(index)
searcher.set_bm25()
searcher.set_rm3()

# Retrieve
runs = []
for topic_id in topics_expanded:
    for query_id in topics_expanded[topic_id]:
        query = topics_expanded[topic_id][query_id]

        hits = searcher.search(query, k=K)

        rows = []
        for rank, hit in enumerate(hits, start=1):
            rows.append((topic_id, 'Q0', hit.docid, rank, hit.score, f'{topic_id}_{query_id}'))
        runs.append(TrecRun.from_list(rows))

run = reciprocal_rank_fusion(runs=runs, depth=K, k=K)
run = pd.DataFrame(data=run.to_numpy(), columns=['q_id', '_1', 'doc_id', '_2', 'score', '_3'])
run = run.astype({'score': 'float'})
run = Run.from_df(run)

run.name = f"BM25+RM3+RFF_{index.split('/')[-1]}"
run.save(f"/home/bd.cardoso/rework/Retrieval/runs/{run.name}.json")

# Evaluate
ndcg = evaluate(Qrels(qrels_3_scale), run, "ndcg@10")
results = evaluate(Qrels(qrels_2_scale_v1), run,
                   ["precision@10", "r-precision", "mrr", "recall@10", "recall@100", "recall@500", "recall@1000",
                    "recall"])
results.update({"ndcg@10": ndcg})
for metric in results:
    results[metric] = round(results[metric], 4)
with open(f"/home/bd.cardoso/rework/Retrieval/runs/{run.name}_metrics.json", "w") as w:
    json.dump(results, w, indent=4)
