import torch
from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer, AutoModel
import sys

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
model = AutoModel.from_pretrained(sys.argv[1])


def getDoc(trial_id, field):
    brief_title = documents[trial_id]["brief_title"] if documents[trial_id]["brief_title"] is not None else ""
    official_title = documents[trial_id]["official_title"] if documents[trial_id][
                                                                  "official_title"] is not None else ""
    brief_summary = documents[trial_id]["brief_summary"] if documents[trial_id]["brief_summary"] is not None else ""
    detailed_description = documents[trial_id]["detailed_description"] if documents[trial_id][
                                                                              "detailed_description"] is not None else ""
    criteria = documents[trial_id]["eligibility"]["criteria"] if documents[trial_id]["eligibility"][
                                                                     "criteria"] is not None else ""
    concat = f"{brief_title} {official_title} {brief_summary} {detailed_description} {criteria}"

    if field == "brief_title":
        return brief_title
    elif field == "official_title":
        return official_title
    elif field == "brief_summary":
        return brief_summary
    elif field == "detailed_description":
        return detailed_description
    elif field == "criteria":
        return criteria
    elif field == "concat":
        return concat


class CTDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data['labels'].size(dim=0)

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.data.items()}


with open('/home/bd.cardoso/my_datasets/csiro_ct/qrels.json', "r") as r:
    qrels = json.load(r)

with open('/home/bd.cardoso/my_datasets/csiro_ct/topics.json', "r") as r:
    topics = json.load(r)

with open('/home/bd.cardoso/my_datasets/csiro_ct/clinical_trials/clinical_trials_gov_2015_12_16.json', "r") as r:
    documents = json.load(r)

data = {"input_ids": [], "attention_mask": [], "labels": []}
for topic in qrels:
    for trial in qrels[topic]:
        q = topics[topic]
        d = getDoc(trial, sys.argv[2])
        input = f"{q} [SEP] {d}"

        encoding = tokenizer(input, padding="max_length", max_length=1024,
                             truncation=True)
        data["input_ids"].append(encoding["input_ids"])
        data["attention_mask"].append(encoding["attention_mask"])

        label = int(qrels[topic][trial])
        if label < 1:
            label = -1.0
        elif label == 1:
            label = 0.0
        else:
            label = 1.0

        data["labels"].append(label)

        if int(qrels[topic][trial]) == 1:
            for i in range(3):
                data["input_ids"].append(encoding["input_ids"])
                data["attention_mask"].append(encoding["attention_mask"])
                data["labels"].append(label)
        elif int(qrels[topic][trial]) == 2:
            for i in range(5):
                data["input_ids"].append(encoding["input_ids"])
                data["attention_mask"].append(encoding["attention_mask"])
                data["labels"].append(label)

data["input_ids"] = torch.tensor(data["input_ids"])
data["attention_mask"] = torch.tensor(data["attention_mask"])
data["labels"] = torch.tensor(data["labels"])
dataset = CTDataset(data)
torch.save(dataset, f"/user/data/bd.cardoso/datasets/regression/{sys.argv[1].replace('/', '-')}_{sys.argv[2]}_1024.pt")
print(len(dataset))
