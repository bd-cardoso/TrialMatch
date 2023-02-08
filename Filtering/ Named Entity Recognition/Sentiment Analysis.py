import ast
import json

import numpy as np
from ranx import Run
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model_name = "pysentimiento/robertuito-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0, return_all_scores=True)


def process_sentiment(intersection, topic_ner, trial_ner, topic, criteria, cont=None):
    if len(intersection) == 0:
        return None
    scores = []
    for word in intersection:
        sentiment = []
        for entity in topic_ner:
            if word == entity["word"]:
                context = topic[entity["start"] - 30:entity["end"] + 30]
                if cont is not None:
                    context += "not"

                sentiment.append(pipe(context))

        topic_labels = {"NEG": np.mean(list(map(lambda x: x[0][0]["score"], sentiment))),
                        #                         "NEU": np.mean(list(map(lambda x: x[0][1]["score"], sentiment))),
                        "POS": np.mean(list(map(lambda x: x[0][2]["score"], sentiment)))}

        sentiment = []
        for entity in trial_ner:
            if word == entity["word"]:
                context = criteria[entity["start"] - 30:entity["end"] + 30]
                sentiment.append(pipe(context))
        trial_labels = {"NEG": np.mean(list(map(lambda x: x[0][0]["score"], sentiment))),
                        #                         "NEU": np.mean(list(map(lambda x: x[0][1]["score"], sentiment))),
                        "POS": np.mean(list(map(lambda x: x[0][2]["score"], sentiment)))}
        neg = abs(topic_labels["NEG"] - trial_labels["NEG"])
        pos = abs(topic_labels["POS"] - trial_labels["POS"])

        score = max(neg, pos)
        scores.append(score)
    return scores


with open(f"/home/bd.cardoso/my_datasets/trec_ct_2021/topics/topics.json", "r") as file:
    topics = json.load(file)

with open(f"/home/bd.cardoso/my_datasets/trec_ct_2021/clinical_trials/clinical_trials_gov_2021_04_27.json",
          "r") as file:
    clinical_trials = json.load(file)

runs = ["/home/bd.cardoso/filtering/sa_data/BM25+RM3+RRF_train.json"]
for run_name in runs:
    for test in ["/home/bd.cardoso/filtering/data_train/BioLinkBERT-base.json",
                 "/home/bd.cardoso/filtering/data_train/BiomedNLP-PubMedBERT-base-uncased-abstract.json"]:
        run = Run.from_file(run_name).to_dict()
        with open(f"{test}", "r") as file:
            ner = ast.literal_eval(json.load(file))

        topics_ids = list(run.keys())
        for topic_id in topics_ids:
            topic = topics[topic_id]
            topic_ner = ner["topics"][topic_id]
            topic_words = list(map(lambda x: x['word'], topic_ner))

            trials_ids = list(run[topic_id].keys())
            for trial_id in trials_ids:
                criteria = clinical_trials[trial_id]['eligibility']["criteria"]
                try:
                    trial_ner = ner["trials"][trial_id]

                    if "inclusion" in trial_ner and "exclusion" in trial_ner:
                        # inclusion
                        trial_words = list(map(lambda x: x['word'], trial_ner["inclusion"]))
                        intersection = list(set(topic_words) & set(trial_words))
                        inclusion_score = process_sentiment(intersection, topic_ner, trial_ner["inclusion"], topic,
                                                            criteria)
                        if inclusion_score is not None:
                            inclusion_score = max(inclusion_score)

                        # exclusion
                        trial_words = list(map(lambda x: x['word'], trial_ner["exclusion"]))
                        intersection = list(set(topic_words) & set(trial_words))
                        exclusion_score = process_sentiment(intersection, topic_ner, trial_ner["exclusion"], topic,
                                                            criteria, cont="NEG")
                        if exclusion_score is not None:
                            exclusion_score = max(exclusion_score)

                        if inclusion_score is not None and exclusion_score is not None:
                            score = max(inclusion_score, exclusion_score)
                        elif inclusion_score is not None:
                            score = inclusion_score
                        elif exclusion_score is not None:
                            score = exclusion_score
                        else:
                            score = None

                    else:
                        trial_words = list(map(lambda x: x['word'], trial_ner))
                        intersection = list(set(topic_words) & set(trial_words))
                        score = process_sentiment(intersection, topic_ner, trial_ner, topic, criteria)
                        if score is not None:
                            score = max(score)

                    if score is not None:
                        run[topic_id][trial_id] -= 1
                except KeyError:
                    pass
        run = Run.from_dict(run)

        run.name = f'sa_{test.split("/")[-1]}'

        run.save(path=f'sa_data/rigid_robertuito_{run.name}.json', kind="json")
        run.save(path=f'sa_data/rigid_robertuito_{run.name}.trec', kind="trec")
