import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


INSTRUCTION = "Classify the following message from a social media platform. It might contain a form of gender-based violence (GBV). Output A if it contains GBV, or B if not."
CHOICES = ["A. GBV", "B. Not GBV"]

POLITICS = {'politicsLeft': 'Left-wing/liberal',
            'politicsRight': 'Right-wing/conservative',
            'politicsCentre': 'Centre',
            'politicsNone': 'None',
            'politicsPrefer': 'Prefer not to say'}

SURVEY_INTRO = "Three scales are used to show the annotator's attitudes, namely the Very Short Authoritarianism (VSA) scale to measure Right Wing Authoritarianism (RWA), the Short Social Dominance Orientation (SSDO) scale to measure Social Dominance Orientation (SDO), and the Brief Hostile Neosexism scale to measure Hostile Neosexism."
SURVEY1 = {'name': 'Very Short Authoritarianism Scale (VSA)',
            'num_Q': 6,
            'scale': ['Very strongly disagree', 'Strongly disagree','Somewhat disagree', 'Slightly disagree',
                    'Unsure or neutral', 'Slightly agree', 'Somewhat agree', 'Strongly agree', 'Very strongly agree'],
            'scale_score': {'Very strongly disagree': 1, 
                      'Strongly disagree': 2,
                      'Somewhat disagree': 3, 
                      'Slightly disagree': 4,
                      'Unsure or neutral': 5, 
                      'Slightly agree': 6, 
                      'Somewhat agree': 7, 
                      'Strongly agree': 8, 
                      'Very strongly agree': 9},
            'Q1': {'statement': 'It’s great that many young people today are prepared to defy authority.',
                    'name': 'defyAuthority',
                    'reverse': True},
            'Q2': {'statement': 'What our country needs most is discipline, with everyone following our leaders in unity.',
                    'name': 'discipline',
                    'reverse': False},
            'Q3': {'statement': 'God’s laws about abortion, pornography, and marriage must be strictly followed before it is too late.',
                    'name': 'GodsLaws',
                    'reverse': False},
            'Q4': {'statement': 'There is nothing wrong with premarital sexual intercourse.',
                    'name': 'premaritalSex',
                    'reverse': True},
            'Q5': {'statement': 'Our society does NOT need tougher Government and stricter Laws.',
                    'name': 'stricterLaws',
                    'reverse': True},
            'Q6': {'statement': 'The facts on crime and the recent public disorders show we have to crack down harder on troublemakers, if we are going to preserve law and order.',
                    'name': 'troublemakers',
                    'reverse': False}}

SURVEY2 = {'name': 'Short Social Dominance Orientation Scale (SSDO)',
            'num_Q': 4,
            'scale': ['Strongly disagree', 'Somewhat disagree', 'Slightly disagree', 'Unsure or neutral',
                        'Slightly agree', 'Somewhat agree', 'Strongly agree'],
            'scale_score': {'Strongly disagree': 1, 
                      'Somewhat disagree': 2, 
                      'Slightly disagree': 3, 
                      'Unsure or neutral': 4,
                      'Slightly agree': 5, 
                      'Somewhat agree': 6, 
                      'Strongly agree': 7},
            'Q1': {'statement': 'In setting priorities, we must consider all societal groups.',
                    'name': 'considerGroups',
                    'reverse': True},
            'Q2': {'statement': 'We should not push for equality of societal groups.',
                    'name': 'notPush',
                    'reverse': False},
            'Q3': {'statement': 'The equality of societal groups should be our goal.',
                    'name': 'equalityGoal',
                    'reverse': True},
            'Q4': {'statement': 'Superior societal groups should dominate inferior groups.',
                    'name': 'considerGroups',
                    'reverse': False}}

SURVEY3 = {'name': 'Brief Hostile Neosexism Scale',
            'num_Q': 6,
            'scale': ['Strongly disagree', 'Somewhat disagree', 'Slightly disagree', 'Unsure or neutral',
                        'Slightly agree', 'Somewhat agree', 'Strongly agree'],
            'scale_score': {'Strongly disagree': 1, 
                      'Somewhat disagree': 2, 
                      'Slightly disagree': 3, 
                      'Unsure or neutral': 4,
                      'Slightly agree': 5, 
                      'Somewhat agree': 6, 
                      'Strongly agree': 7},
            'Q1': {'statement': 'Some of the demands of the feminist movement seem to me to be a bit exaggerated.',
                    'name': 'feminismMovement',
                    'reverse': False},
            'Q2': {'statement': 'I sometimes feel that our society pays too much attention to the rights of certain minorities.',
                    'name': 'minorityAttention',
                    'reverse': False},
            'Q3': {'statement': 'In the name of equality, many women try to gain certain privileges.',
                    'name': 'womenPrivilege',
                    'reverse': False},
            'Q4': {'statement': 'Many women interpret innocent comments and actions as sexist.',
                    'name': 'innocentAction',
                    'reverse': False},
            'Q5': {'statement': 'Women are easily offended.',
                    'name': 'womenOffended',
                    'reverse': False},
            'Q6': {'statement': 'Women exaggerate the problems they suffer because they are women.',
                    'name': 'exaggerateProblem',
                    'reverse': False}}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def load_original_dataset() -> dict:
    train_val = pd.read_csv("data/edos-guest-train-600-anno.tsv", sep="\t", header=0)
    train, val = train_test_split(train_val, test_size=0.2, random_state=42, shuffle=True)
    test = pd.read_csv("data/edos-guest-test-1000-anno.tsv", sep="\t", header=0)
    return {"train": train, "val": val, "test": test}


def load_survey_dataset() -> pd.DataFrame:
    survey = pd.read_csv("data/survey_v3_batch_results.csv", sep=",", header=0)
    return survey


def convert_hs_label(ds_name,fm_name):
    if ds_name == "edos":
        label1 = "sexist"
        label2 = "not sexist"

    elif ds_name == "guest":
        label1 = "Misogynistic"
        label2 = "Nonmisogynistic"

    # if fm_name == "alpaca-label":
    #     return {label1: "GBV", label2: "Not GBV"}
    if fm_name == "alpaca-option2":
        return {label1: CHOICES[0], label2: CHOICES[1]}
    else:
        raise ValueError("invalid alpaca format name (alpaca-option2 is allowed).")


def get_politic_answer(politics, worker):
    for key in politics.keys():
        name = "Answer." + str(key) + "." + politics[key]
        if worker[name]:
            if politics[key] == "Prefer not to say":
                answer = False
            else:
                answer = politics[key].lower()
    return answer


def get_survey_answer(survey, worker):
    n_scale = len(survey["scale"])
    for i in range(survey["num_Q"]):
        question_id = 'Q' + str(i+1)
        for j in range(n_scale):
            q_name = survey[question_id]["name"]
            name = f"Answer.{q_name}{j+1}.{j+1}"
            if worker[name]:
                answer = survey["scale"][-j] if survey[question_id]["reverse"] else survey["scale"][j]
        survey[question_id]["answer"] = answer
        survey[question_id]["score"] = (
                    len(survey["scale"])+1-survey["scale_score"][answer] if survey[question_id]["reverse"] 
                    else survey["scale_score"][answer]
                    )
    return survey


def get_survey_range(worker_survey):
    # 'survey1': {'name': 'Very Short Authoritarianism Scale (VSA)',
    survey1 = worker_survey["survey1"]
    score1 = 0
    for i in range(survey1["num_Q"]):
        question_id = 'Q' + str(i+1)
        score1 += survey1[question_id]["score"]
    if score1 >= 6 and score1 <= 15:
        range1 = "very low right wing authoritarianism"
    elif score1 >= 16 and score1 <= 25:
        range1 = "low right wing authoritarianism"
    elif score1 >= 26 and score1 <= 35:
        range1 = "moderate right wing authoritarianism"
    elif score1 >= 36 and score1 <= 45:
        range1 = "high right wing authoritarianism"
    elif score1 >= 46 and score1 <= 54:
        range1 = "very high right wing authoritarianism"
    
    # 'survey2': {'name': 'Short Social Dominance Orientation Scale (SSDO)',
    survey2 = worker_survey["survey2"]
    score2 = 0
    for i in range(survey2["num_Q"]):
        question_id = 'Q' + str(i+1)
        score2 += survey2[question_id]["score"]
    if score2 >= 4 and score2 <= 10:
        range2 = "low social dominance orientation"
    elif score2 >= 11 and score2 <= 17:
        range2 = "moderate social dominance orientation"
    elif score2 >= 18 and score2 <= 28:
        range2 = "high social dominance orientation"
    
    # 'survey3': {'name': 'Brief Hostile Neosexism Scale',
    survey3 = worker_survey["survey3"]
    score3 = 0
    for i in range(survey3["num_Q"]):
        question_id = 'Q' + str(i+1)
        score3 += survey3[question_id]["score"]
    if score3 >= 6 and score3 <= 14:
        range3 = "low hostile neosexism"
    elif score3 >= 15 and score3 <= 28:
        range3 = "moderate hostile neosexism"
    elif score3 >= 29 and score3 <= 42:
        range3 = "high hostile neosexism"

    worker_survey["survey1"]["overall_score"] = score1 
    worker_survey["survey2"]["overall_score"] = score2 
    worker_survey["survey3"]["overall_score"] = score3 

    worker_survey["survey1"]["range"] = range1 
    worker_survey["survey2"]["range"] = range2 
    worker_survey["survey3"]["range"] = range3 
    return worker_survey


def format_worker_info(worker):
    info = {}
    info["age"] = worker["Answer.age"]
    info["ethnicity"] = worker["Answer.ethnicity"].lower()
    info["gender"] = worker["Answer.gender"].lower()
    info["sexual orientation"] = worker["Answer.sexualOrientation"].lower() if type(worker["Answer.sexualOrientation"])==str else False
    info["politics"] = get_politic_answer(POLITICS, worker).lower() if get_politic_answer(POLITICS, worker) else get_politic_answer(POLITICS, worker)

    info["survey1"] = get_survey_answer(SURVEY1, worker)
    info["survey2"] = get_survey_answer(SURVEY2, worker)
    info["survey3"] = get_survey_answer(SURVEY3, worker)
    info = get_survey_range(info)
    return info
    

def generate_anno_intro(info):

    age = info["age"]
    ethnicity = info["ethnicity"]
    gender = info["gender"]
    prompt = f"This annotator is a {age}-year-old {ethnicity} {gender}."

    sexual_orientation = info["sexual orientation"]
    politics = info["politics"]
    if sexual_orientation and politics:
        prompt = f"{prompt[:-1]}, who is {sexual_orientation} and {politics} politics."
    elif sexual_orientation and not politics:
        prompt = f"{prompt[:-1]}, who is {sexual_orientation}."
    elif sexual_orientation and not politics:
        prompt = f"{prompt[:-1]}, who is {politics} politics."

    prompt = f"{prompt}\n{SURVEY_INTRO} The following are questions and annotator's answers from each scale.\n"

    s1 = info["survey1"]["name"]
    prompt = f"{prompt}Scale 1: {s1}\n"
    for i in range(info["survey1"]["num_Q"]):
        question = info["survey1"]["Q"+str(i+1)]["statement"]
        answer = info["survey1"]["Q"+str(i+1)]["answer"]
        prompt = f"{prompt}Statement {i+1}: {question}\nAnswer: {answer}"

    s2 = info["survey2"]["name"]
    prompt = f"{prompt}Scale 2: {s2}\n"
    for i in range(info["survey2"]["num_Q"]):
        question = info["survey2"]["Q"+str(i+1)]["statement"]
        answer = info["survey2"]["Q"+str(i+1)]["answer"]
        prompt = f"{prompt}Statement {i+1}: {question}\nAnswer: {answer}"

    s3 = info["survey3"]["name"]
    prompt = f"{prompt}Scale 3: {s3}\n"
    for i in range(info["survey3"]["num_Q"]):
        question = info["survey3"]["Q"+str(i+1)]["statement"]
        answer = info["survey3"]["Q"+str(i+1)]["answer"]
        prompt = f"{prompt}Statement {i+1}: {question}\nAnswer: {answer}\n"

    return prompt   


def generate_anno_intro_short(info):
    age = info["age"]
    ethnicity = info["ethnicity"]
    gender = info["gender"]
    prompt = f"This annotator is a {age}-year-old {ethnicity} {gender}."

    sexual_orientation = info["sexual orientation"]
    politics = info["politics"]
    if sexual_orientation and politics:
        prompt = f"{prompt[:-1]}, who is {sexual_orientation} and {politics} politics."
    elif sexual_orientation and not politics:
        prompt = f"{prompt[:-1]}, who is {sexual_orientation}."
    elif sexual_orientation and not politics:
        prompt = f"{prompt[:-1]}, who is {politics} politics."

    range1 = info["survey1"]["range"]
    range2 = info["survey2"]["range"]
    range3 = info["survey3"]["range"]
    prompt = f"{prompt}\n{SURVEY_INTRO} This worker is {range1}, {range2}, and {range3}."
    return prompt


def format_data(df,fm_name,workers):
    label_dict = convert_hs_label("edos", fm_name)
    formatted_data = [
                    {
                    "instruction": INSTRUCTION,
                    "input": row_dict["text"],
                    "choices": CHOICES,
                    "output": label_dict[row_dict["label_1"]],
                    "output_anno": label_dict[row_dict["amt_label"]],
                    "anno_id": row_dict["amt_id"],
                    "anno_info": workers[row_dict["amt_id"]]["info"],
                    "anno_prompt": workers[row_dict["amt_id"]]["prompt"],
                    "anno_short_prompt": workers[row_dict["amt_id"]]["short_prompt"],
                    "id": row_dict["id"],
                    }
                    for row_dict in df.to_dict(orient="records")
                ]
    return formatted_data


def main(
    datasets: str = "mix-single", 
    formats: str = "alpaca-option2", #"alpaca-label,alpaca-option,bert",
    ) -> None:
    """Process original datasets into json files.

    Args:
        datasets: The datasets to use as a comma separated string
        formats: Data formats to be processed - a comma separated string
        save_samples: Determine if sample data is saved
        n_samples: Number of samples to be saved
    """

    survey = load_survey_dataset()
    workers = {}
    save_workers = [] # save useful work info as a single table
    for i in range(survey.shape[0]):
        worker = survey.iloc[i,:]
        info = format_worker_info(worker)
        # workers[worker["WorkerId"]] = {}
        # workers[worker["WorkerId"]]["info"] = info
        # workers[worker["WorkerId"]]["prompt"] = generate_anno_intro(info)
        # workers[worker["WorkerId"]]["short_prompt"] = generate_anno_intro_short(info)
        
        save_worker = {}
        save_worker["amt_id"] = worker["WorkerId"]
        save_worker["age"] = info["age"]
        save_worker["ethnicity"] = info["ethnicity"]
        save_worker["gender"] = info["gender"]
        save_worker["sexual_orientation"] = info["sexual orientation"]
        save_worker["politics"] = info["politics"]
        save_worker["Very Short Authoritarianism Scale (VSA)"] = info["survey1"]["range"]
        save_worker["Short Social Dominance Orientation Scale (SSDO)"] = info["survey2"]["range"]
        save_worker["Brief Hostile Neosexism Scale"] = info["survey3"]["range"]
        save_workers.append(save_worker)
    
    
    save_workers_df = pd.DataFrame(save_workers)
    save_workers_df.to_csv("data/annotator-info.tsv", sep="\t", header=True, index=False)
    print("save worker info to data/annotator-info.tsv")
        
        
    # for ds_name in datasets.split(","):
    #     for fm_name in formats.split(","):
    #         data_splits = load_original_dataset()
    #         save_dir = "data/" + ds_name + "-" + fm_name 
    #         os.makedirs(save_dir, exist_ok=True)

    #         for key in data_splits.keys():
    #             formatted_data = format_data(data_splits[key], fm_name, workers)
    #             # print(formatted_data)
    #             filename =  str(key)+".json"
    #             save_file = Path(save_dir, filename)
    #             with open(save_file, "w") as f:
    #                 json.dump(formatted_data, f, cls=NumpyEncoder)

    #             print(f"Saved to {str(save_file)}")


if __name__ == "__main__":
    
    main()