from ranx import Run, Qrels, evaluate
import json
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import sys

run = Run.from_file("/home/bd.cardoso/rework/Retrieval/runs/BM25+RM3+RFF_all_free_text_fields.json").to_dict()
checkpoint = sys.argv[1]

model = BertForSequenceClassification.from_pretrained(checkpoint)
tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

with open(f"/home/bd.cardoso/my_datasets/trec_ct_2021/topics/topics.json", "r") as file:
    topics = json.load(file)

with open(f"/home/bd.cardoso/my_datasets/trec_ct_2021/clinical_trials/clinical_trials_gov_2021_04_27.json",
          "r") as file:
    documents = json.load(file)

for topic_id in run:
    topic = topics[topic_id]
    for trial_id in run[topic_id]:
        brief_title = documents[trial_id]["brief_title"] if documents[trial_id]["brief_title"] is not None else ""
        official_title = documents[trial_id]["official_title"] if documents[trial_id][
                                                                      "official_title"] is not None else ""
        brief_summary = documents[trial_id]["brief_summary"] if documents[trial_id]["brief_summary"] is not None else ""
        detailed_description = documents[trial_id]["detailed_description"] if documents[trial_id][
                                                                                  "detailed_description"] is not None else ""
        criteria = documents[trial_id]["eligibility"]["criteria"] if documents[trial_id]["eligibility"][
                                                                         "criteria"] is not None else ""
        concat = f"{brief_title} {official_title} {brief_summary} {detailed_description} {criteria}"

        encoding = tokenizer(f'{topic} [SEP] {concat}', return_tensors="pt", padding="max_length",
                             max_length=model.config.max_position_embeddings,
                             truncation=True).to(device)

        model.eval()
        with torch.no_grad():
            logits = model(**encoding).logits

        run[topic_id][trial_id] = float(logits[0][0].cpu())

run = Run.from_dict(run)
run.save(
    path=f"/home/bd.cardoso/rework/Zero-Shot/AutoModelForSequenceClassification/runs/{checkpoint.replace('/', '_')}_concat.json",
    kind="json")

with open("/home/bd.cardoso/my_datasets/trec_ct_2021/qrels/qrels_3_scale.json", "r") as r:
    qrels_3_scale = json.load(r)

with open("/home/bd.cardoso/my_datasets/trec_ct_2021/qrels/qrels_2_scale_v1.json", "r") as r:
    qrels_2_scale_v1 = json.load(r)

# Evaluate
ndcg = evaluate(Qrels(qrels_3_scale), run, "ndcg@10")
results = evaluate(Qrels(qrels_2_scale_v1), run,
                   ["precision@10", "r-precision", "mrr", "recall@10", "recall@100", "recall@500", "recall@1000",
                    "recall"])
results.update({"ndcg@10": ndcg})
for metric in results:
    results[metric] = round(results[metric], 4)
with open(
        f"/home/bd.cardoso/rework/Zero-Shot/AutoModelForSequenceClassification/runs/my/{checkpoint.replace('/', '_')}_concat_metrics.json",
        "w") as w:
    json.dump(results, w, indent=4)
