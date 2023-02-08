import os
import re
import xml.etree.ElementTree as ET
import json

import clinical_trials_gov_utils as ct_utils

data = {}

count = 0
data_dir = "D:/Thesis/datasets/trec_ct"
for part in os.listdir(data_dir):
    for folder in os.listdir(f'{data_dir}/{part}'):
        if folder == 'Contents.txt':
            continue
        for file in os.listdir(f'{data_dir}/{part}/{folder}'):
            count += 1
            print(f"\r{count}\\375580", end="")

            path = f'{data_dir}/{part}/{folder}/{file}'
            ct_id = re.findall(r'NCT\d{8}', path)[0]

            root = ET.parse(path).getroot()

            brief_title = ct_utils.remove_whitespaces_except_one_space_from_field(
                ct_utils.get_brief_title(root))

            if brief_title == "[Trial of device that is not approved or cleared by the U.S. FDA]":
                continue

            official_title = ct_utils.remove_whitespaces_except_one_space_from_field(
                ct_utils.get_official_title(root))

            brief_summary = ct_utils.remove_whitespaces_except_one_space_from_field(
                ct_utils.get_brief_summary(root))

            detailed_description = ct_utils.remove_whitespaces_except_one_space_from_field(
                ct_utils.get_detailed_description(root))

            conditions = ct_utils.get_conditions(root)

            eligibility_study_pop = ct_utils.remove_whitespaces_except_one_space_from_field(
                ct_utils.get_eligibility_study_pop(root))

            eligibility_criteria = ct_utils.remove_whitespaces_except_one_space_from_field(
                ct_utils.get_eligibility_criteria(root))

            eligibility_gender = ct_utils.get_eligibility_gender(root)
            eligibility_minimum_age = ct_utils.get_eligibility_minimum_age(root)
            eligibility_maximum_age = ct_utils.get_eligibility_maximum_age(root)
            eligibility_healthy_volunteers = ct_utils.get_eligibility_healthy_volunteers(root)

            ct = {
                "brief_title": brief_title,
                "official_title": official_title,
                "brief_summary": brief_summary,
                "detailed_description": detailed_description,

                "condition": conditions,

                "eligibility": {
                    "study_pop": eligibility_study_pop,
                    "criteria": eligibility_criteria,
                    "gender": eligibility_gender,
                    "minimum_age": eligibility_minimum_age,
                    "maximum_age": eligibility_maximum_age,
                    "healthy_volunteers": eligibility_healthy_volunteers
                }
            }

            data[ct_id] = ct

with open("D:/Thesis/my_datasets/trec_ct_2021/clinical_trials/clinical_trials_gov_2021_04_27.json", "w") as file:
    json.dump(data, file, indent=4)
