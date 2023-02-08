import xml.etree.ElementTree as ET
import json

import utils.clinical_trials_gov_utils as ct_utils

data = {}

root = ET.parse("D:/Thesis/datasets/trec-ct-2021/topics2021.xml").getroot()
for element in root.findall('topic'):
    topic_id = element.attrib['number']
    description = ct_utils.remove_whitespaces_except_one_space_from_field(element.text)
    data[topic_id] = description

with open("D:/Thesis/my_datasets/trec_ct_2021/topics.json", "w") as file:
    json.dump(data, file, indent=4)
