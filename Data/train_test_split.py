import pandas as pd
import json

df = pd.read_csv('/datasets/trec-ct-2021/2021-qrels.txt', header=None, delimiter=' ')
df.columns = ['topic', '_', 'ct', 'rel']
df = df.drop(columns=['_'])
data = df.drop(columns=['ct'])

number_rels = data.groupby(['topic']).count().reset_index().rename(columns={'rel': 'number of examples'})
sum_rels = data.groupby(['topic']).sum().reset_index().rename(columns={'rel': 'number of positive examples'})

result = number_rels.set_index('topic').join(sum_rels.set_index('topic')).reset_index()
result = result.sort_values(by=['number of positive examples'])

l = []
b = True
for i in range(75):
    if b:
        l.append("train")
    else:
        l.append("test")
    b = not b

ser = pd.Series(l)
result["train/test"] = ser.values

train = result[result["train/test"] == "train"].sort_values(by=['topic'])
test = result[result["train/test"] == "test"].sort_values(by=['topic'])

train_hist = train.hist(column=["number of examples", "number of positive examples"], bins=6)
train_hist[0][0].get_figure().savefig("/my_datasets/trec_ct_2021/train_test_split/train_50.png")
test_hist = test.hist(column=["number of examples", "number of positive examples"], bins=6)
test_hist[0][0].get_figure().savefig("/my_datasets/trec_ct_2021/train_test_split/test_50.png")

with open("/my_datasets/trec_ct_2021/train_test_split/train_50.json", "w") as file:
    json.dump(train['topic'].tolist(), file, indent=4)

with open("/my_datasets/trec_ct_2021/train_test_split/test_50.json", "w") as file:
    json.dump(test['topic'].tolist(), file, indent=4)

train = df[df["topic"].isin(train["topic"])]
test = df[df["topic"].isin(test["topic"])]

train.to_csv("/my_datasets/trec_ct_2021/train_test_split/train_50.csv", index=False)
test.to_csv("/my_datasets/trec_ct_2021/train_test_split/test_50.csv", index=False)
