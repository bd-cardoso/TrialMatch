import json

data = {}

with open("D:/Thesis/datasets/trec-ct-2021/2021-qrels.txt", "r") as file:
    lines = file.readlines()

for line in lines:
    line = line.strip().split(' ')
    query = line[0]
    document = line[2]
    relevance = int(line[3])

    if query not in data:
        data[query] = {}
    data[query][document] = relevance

with open("D:/Thesis/my_datasets/trec_ct_2021/qrels.json", "w") as file:
    json.dump(data, file, indent=4)
