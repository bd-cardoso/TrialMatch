import ast
import json

import numpy as np
from ranx import Run
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# https://huggingface.co/bvanaken/clinical-assertion-negation-bert
tokenizer = AutoTokenizer.from_pretrained("bvanaken/clinical-assertion-negation-bert")
model = AutoModelForSequenceClassification.from_pretrained("bvanaken/clinical-assertion-negation-bert")


def process_sentiment(intersection, topic_ner, trial_ner, topic, criteria, cont=None):
    if len(intersection) == 0:
        return None

    scores = []
    for word in intersection:
        sentiment = []
        for entity in topic_ner:
            if word == entity["word"]:
                context = topic[entity["start"] - 30:entity["end"] + 30]
                context = context.lower().split(word)
                if len(context) != 1:
                    context = context[0] + "[entity]" + word + "[entity]" + context[1]
                else:
                    context = context[0]

                tokenized_input = tokenizer(context, return_tensors="pt")
                output = model(**tokenized_input)
                predicted_label = (output.logits.detach().numpy())[0]  ## 1 == ABSENT
                predicted_label = np.exp(predicted_label) / np.sum(np.exp(predicted_label))
                sentiment.append(predicted_label)

                topic_labels = {"NEG": np.mean(list(map(lambda x: x[0], sentiment))),
                                "POS": np.mean(list(map(lambda x: x[1], sentiment)))}

        sentiment = []
        for entity in trial_ner:
            if word == entity["word"]:
                context = criteria[entity["start"] - 30:entity["end"] + 30]
                if cont is not None:
                    context = "not " + context
                context = topic[entity["start"] - 30:entity["end"] + 30]
                context = context.lower().split(word)
                if len(context) != 1:
                    context = context[0] + "[entity]" + word + "[entity]" + context[1]
                else:
                    context = context[0]
                tokenized_input = tokenizer(context, return_tensors="pt")
                output = model(**tokenized_input)
                predicted_label = (output.logits.detach().numpy())[0]  ## 1 == ABSENT
                predicted_label = np.exp(predicted_label) / np.sum(np.exp(predicted_label))
                sentiment.append(predicted_label)

                trial_labels = {"NEG": np.mean(list(map(lambda x: x[0], sentiment))),
                                "POS": np.mean(list(map(lambda x: x[1], sentiment)))}
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
    #     for test in ["/home/bd.cardoso/filtering/data_train/BC2GM_hf_BioLinkBERT-base.json"]:
    #     for test in ["/home/bd.cardoso/filtering/data_train/BC5CDR-chem_hf_BioLinkBERT-base.json"]:
    #     for test in ["/home/bd.cardoso/filtering/data_train/JNLPBA_hf_BioLinkBERT-base.json"]:
    #     for test in ["/home/bd.cardoso/filtering/data_train/BioLinkBERT-base.json"]:
    for test in ["/home/bd.cardoso/filtering/data_train/BiomedNLP-PubMedBERT-base-uncased-abstract.json"]:
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

        run.name = f'sa_{test.split("/")[-1]}_clinical-assertion-negation-bert'

        run.save(path=f'assertion/{run.name}.json', kind="json")
        run.save(path=f'assertion/{run.name}.trec', kind="trec")
