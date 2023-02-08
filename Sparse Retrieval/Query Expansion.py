import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
from pprint import pprint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-large-msmarco')
model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-large-msmarco')
model.to(device)

with open(f"/home/bd.cardoso/my_datasets/trec_ct_2021/topics/topics.json", "r") as file:
    topics = json.load(file)

topics_expanded = {}
for topic_id in topics:
    topics_expanded[topic_id] = {}
    topics_expanded[topic_id][0] = topics[topic_id]

    input_ids = tokenizer.encode(topics[topic_id], return_tensors='pt').to(device)
    outputs = model.generate(
        input_ids=input_ids,
        max_length=64,
        do_sample=True,
        top_k=10,
        num_return_sequences=40)

    for i in range(40):
        topics_expanded[topic_id][i + 1] = tokenizer.decode(outputs[i], skip_special_tokens=True)

with open(f"/home/bd.cardoso/my_datasets/trec_ct_2021/topics/topics_expanded.json", "w") as file:
    json.dump(topics_expanded, file, indent=4)
