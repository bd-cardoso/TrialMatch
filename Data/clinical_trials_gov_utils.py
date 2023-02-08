import re
import string


def get_brief_title(root):
    try:
        return root.find('brief_title').text
    except AttributeError:
        return None


def get_official_title(root):
    try:
        return root.find('official_title').text
    except AttributeError:
        return None


def get_brief_summary(root):
    try:
        return root.find('brief_summary').findtext('textblock')
    except AttributeError:
        return None


def get_detailed_description(root):
    try:
        return root.find('detailed_description').findtext('textblock')
    except AttributeError:
        return None


def get_conditions(root):
    try:
        _ = root.find('condition').text  # Just to throw exception
        return list(map(lambda e: e.text, root.findall('condition')))
    except AttributeError:
        return None


def get_eligibility_study_pop(root):
    try:
        return root.find('eligibility').find('study_pop').findtext('textblock')
    except AttributeError:
        return None


def get_eligibility_criteria(root):
    try:
        return root.find('eligibility').find('criteria').findtext('textblock')
    except AttributeError:
        return None


def get_eligibility_gender(root):
    try:
        return root.find('eligibility').find('gender').text
    except AttributeError:
        return None


def get_eligibility_minimum_age(root):
    try:
        return root.find('eligibility').find('minimum_age').text
    except AttributeError:
        return None


def get_eligibility_maximum_age(root):
    try:
        return root.find('eligibility').find('maximum_age').text
    except AttributeError:
        return None


def get_eligibility_healthy_volunteers(root):
    try:
        return root.find('eligibility').find('healthy_volunteers').text
    except AttributeError:
        return None


def is_eligibility_criteria_semi_structured(criteria):
    if len(re.findall(r'Inclusion Criteria:[\s\S]*Exclusion Criteria:', criteria, flags=re.MULTILINE | re.DOTALL)) == 0:
        return False
    else:
        return True


def is_field_applicable(field):
    field = remove_whitespaces_except_one_space_from_field(field)
    if field == 'N/A':
        return False
    else:
        return True


def remove_whitespaces_except_one_space_from_field(field):
    if field is None:
        return None

    whitespace_except_space = string.whitespace.replace(' ', '')

    field.strip(whitespace_except_space)
    field = ' '.join(field.split())
    return field


def concatenate_fields(fields):
    return ' '.join(filter(None, fields))


def space_tokenizer(field):
    return field.split(' ')
