import json

import utils.clinical_trials_gov_utils as ct_utils

brief_title = []
official_title = []
brief_summary = []
detailed_description = []
eligibility_study_pop = []
eligibility_criteria = []
all_free_text_fields = []

with open("D:/Thesis/my_datasets/trec_ct_2021/clinical_trials/clinical_trials_gov_2021_04_27.json", "r") as file:
    data = json.load(file)

count = 0
for ct in data:
    count += 1
    print(f"\r{count}\\374739", end="")

    brief_title.append({"id": ct,
                        "contents": data[ct]["brief_title"] if data[ct]["brief_title"] is not None else ""})

    official_title.append({"id": ct,
                           "contents": data[ct]["official_title"] if data[ct]["official_title"] is not None else ""})

    brief_summary.append({"id": ct,
                          "contents": data[ct]["brief_summary"] if data[ct]["brief_summary"] is not None else ""})

    detailed_description.append({"id": ct,
                                 "contents": data[ct]["detailed_description"] if data[ct][
                                                                                     "detailed_description"] is not None else ""})

    eligibility_study_pop.append({"id": ct,
                                  "contents": data[ct]["eligibility"]["study_pop"] if data[ct]["eligibility"][
                                                                                          "study_pop"] is not None else ""})

    eligibility_criteria.append({"id": ct,
                                 "contents": data[ct]["eligibility"]["criteria"] if data[ct]["eligibility"][
                                                                                        "criteria"] is not None else ""})

    concatenated_free_text_fields = ct_utils.concatenate_fields([
        data[ct]["brief_title"],
        data[ct]["official_title"],
        data[ct]["brief_summary"],
        data[ct]["detailed_description"],
        data[ct]["eligibility"]["study_pop"],
        data[ct]["eligibility"]["criteria"]
    ])

    all_free_text_fields.append({"id": ct,
                                 "contents": concatenated_free_text_fields})

with open("D:/Thesis/my_datasets/trec_ct_2021/clinical_trials/brief_title/documents.json", "w") as file:
    json.dump(brief_title, file, indent=4)

with open("D:/Thesis/my_datasets/trec_ct_2021/clinical_trials/official_title/documents.json", "w") as file:
    json.dump(official_title, file, indent=4)

with open("D:/Thesis/my_datasets/trec_ct_2021/clinical_trials/brief_summary/documents.json", "w") as file:
    json.dump(brief_summary, file, indent=4)

with open("D:/Thesis/my_datasets/trec_ct_2021/clinical_trials/detailed_description/documents.json", "w") as file:
    json.dump(detailed_description, file, indent=4)

with open("D:/Thesis/my_datasets/trec_ct_2021/clinical_trials/eligibility_study_pop/documents.json", "w") as file:
    json.dump(eligibility_study_pop, file, indent=4)

with open("D:/Thesis/my_datasets/trec_ct_2021/clinical_trials/eligibility_criteria/documents.json", "w") as file:
    json.dump(eligibility_criteria, file, indent=4)

with open("D:/Thesis/my_datasets/trec_ct_2021/clinical_trials/all_free_text_fields/documents.json", "w") as file:
    json.dump(all_free_text_fields, file, indent=4)
