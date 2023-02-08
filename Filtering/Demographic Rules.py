from ranx import Run, evaluate, Qrels
import json
import re

with open(f"/home/bd.cardoso/my_datasets/trec_ct_2021/topics/topics.json", "r") as file:
    topics = json.load(file)

with open(f"/home/bd.cardoso/my_datasets/trec_ct_2021/clinical_trials/clinical_trials_gov_2021_04_27.json",
          "r") as file:
    clinical_trials = json.load(file)

run = Run.from_file("/home/bd.cardoso/rework/Retrieval/runs/BM25+RM3+RFF_all_free_text_fields.json").to_dict()

topics_ids = list(run.keys())
for topic_id in topics_ids:
    m = re.search(
        r"\d+ *(yo|year old|-year-old|year-old|y/o|-year old|year|-day-old|months old)?"
        r"(( *(\w*|African-American) *){,6}(man|woman|female|gentleman|male|boy|girl|F|M))?",
        topics[topic_id])[0]
    if "-day-old" not in m and "months old" not in m:
        age = int(re.search(r"\d+", m)[0])
    else:
        age = 0
    if "woman" in m or "female" in m or "girl" in m or "F" in m:
        gender = "Female"
    elif "man" in m or "gentleman" in m or "male" in m or "boy" in m or "M" in m:
        gender = "Male"
    else:
        gender = None

    trials_ids = list(run[topic_id].keys())
    for trial_id in trials_ids:
        eligible = True
        if clinical_trials[trial_id]['eligibility']['gender'] == 'All':
            pass
        elif clinical_trials[trial_id]['eligibility']['gender'] != gender:
            eligible = False

        maximum_age = clinical_trials[trial_id]['eligibility']['maximum_age']
        if maximum_age != "N/A" and maximum_age is not None:
            maximum_age = int(re.search(r"\d+", maximum_age)[0])
            if age >= maximum_age:
                eligible = False

        minimum_age = clinical_trials[trial_id]['eligibility']['minimum_age']
        if minimum_age != "N/A" and minimum_age is not None:
            minimum_age = int(re.search(r"\d+", minimum_age)[0])
            if age <= minimum_age:
                eligible = False

        if not eligible:
            run[topic_id][trial_id] -= 1

run = Run.from_dict(run)
run.save(
    path=f"/home/bd.cardoso/rework/Retrieval/runs/regex.json",
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
        f"/home/bd.cardoso/rework/Retrieval/runs/regex_metrics.json",
        "w") as w:
    json.dump(results, w, indent=4)
