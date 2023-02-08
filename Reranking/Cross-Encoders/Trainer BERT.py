import torch
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer, BertForSequenceClassification, LongformerForSequenceClassification
import wandb
import sys


class CTDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data['labels'].size(dim=0)

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.data.items()}


data_dir = "/user/data/bd.cardoso/datasets/regression"
aux = ["brief_title",
       "official_title",
       "brief_summary",
       "detailed_description",
       "criteria",
       "concat"]

# # aux = ["criteria"]
# aux = ["concat"]

for x in aux:
    dataset = f"{data_dir}/{sys.argv[1].replace('/', '-')}_{x}_1024.pt"

    model = BertForSequenceClassification.from_pretrained(sys.argv[1], num_labels=1)
    wandb.init(project=f"{sys.argv[1].replace('/', '-')}_{x}")
    train_dataset = torch.load(dataset)

    args = TrainingArguments(output_dir=f"/user/data/bd.cardoso/long/{sys.argv[1].replace('/', '-')}_{x}_1024",
                             overwrite_output_dir=True,
                             do_train=True,
                             gradient_accumulation_steps=4,
                             save_strategy="epoch",
                             num_train_epochs=5,
                             logging_steps=1,
                             logging_strategy="epoch",
                             logging_first_step=True,
                             report_to="wandb",
                             per_device_train_batch_size=8,
                             fp16=True)

    trainer = Trainer(model=model, train_dataset=train_dataset, args=args)

    trainer.train()
    trainer.save_model()
