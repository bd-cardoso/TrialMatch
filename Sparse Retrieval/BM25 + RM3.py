import json
from pyserini.search.lucene import LuceneSearcher
from ranx import Qrels, Run, evaluate

K = 1000

index = "/home/bd.cardoso/pyserini/indexes/trec_ct_2021/all_free_text_fields"

with open("/home/bd.cardoso/my_datasets/trec_ct_2021/topics/topics.json", "r") as r:
    topics = json.load(r)

with open("/home/bd.cardoso/my_datasets/trec_ct_2021/qrels/qrels_3_scale.json", "r") as r:
    qrels_3_scale = json.load(r)

with open("/home/bd.cardoso/my_datasets/trec_ct_2021/qrels/qrels_2_scale_v1.json", "r") as r:
    qrels_2_scale_v1 = json.load(r)

run_dict = {}
searcher = LuceneSearcher(index)
searcher.set_bm25()
searcher.set_rm3()

# Retrieve
for topic_id in topics:
    if topic_id not in run_dict:
        run_dict[topic_id] = {}

    hits = searcher.search(topics[topic_id], k=K)
    for hit in hits:
        run_dict[topic_id][hit.docid] = hit.score

run = Run(run_dict, name=f"BM25+RM3_{index.split('/')[-1]}")
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
