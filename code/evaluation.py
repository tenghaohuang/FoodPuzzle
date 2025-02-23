from typing import Union
import os
import pickle
import resource
import sys
from DSP_functions import predict_functional_groups
from tqdm import tqdm
import dspy
from config import Config
from utils import predict_food_category, predict_functional_group
from IPython import embed
import sys
import ast
# sys.setrecursionlimit(1000000)

def eval_task1(results:Union[str, dict]='baseline_results_100-gpt-3.5.p'):
    """
    Args:
        result_file Union[str, dict]: Path to the pickle file containing the prediction results or dictionary

    Returns:
        Tuple of:
        - Accuracy
        - List of updated results with hit.
    """
    if isinstance(results, str):
        results = pickle.load(open(results,"rb"))
    task1 = results['Task1']

    flavordb = pickle.load(open(Config.db_address,"rb"))
    fid2food = flavordb['id2food']
    
    food2category = {}
    for i in fid2food.values():
        food2category[i['entity_alias_readable']] = i['category'].split("-")[0]
    
    macro_categories = [ # 23
        'cereal',
        'plant',
        'seed',
        'flower',
        'animalproduct',
        'additive',
        'fishseafood',
        'dairy',
        'fruit',
        'bakery',
        'dish',
        'nutseed',
        'vegetable',
        'meat',
        'cerealcrop',
        'herb',
        'essentialoil',
        'fungus',
        'spice',
        'Vegetable',
        'beverage',
        'plantderivative']
    
    correct = 0
    fail = 0
    results = []
    print("\n== Task 1 Eval ==\n")
    for item in tqdm(task1):
        gt_food = item['actual_food']
        gt_cate = food2category[gt_food]
        # gt_cate = predict_food_category(gt_food, macro_categories)
        predicted_food = item['predicted_food']

        if type(predicted_food) != str:
            predicted_food = 'error'
        else:
            predicted_food = predicted_food.replace('[','').replace(']','').replace("*", '')

        if 'error' in predicted_food.lower():
            print("** LLM response error")
            predicted_cate = predicted_food
        else:
            predicted_cate = predict_food_category(predicted_food, macro_categories, api_key=Config.OPENAI_API_KEY)
            print(f"* Ground truth category: {gt_cate} \n* Predicted category: {predicted_cate}")
        
        if not isinstance(predicted_cate, str):
            print("****** predict food category error ******")
            fail += 1
            continue
        
        hit=0
        if gt_cate.strip().lower() == predicted_cate.strip().lower():
            correct += 1
            hit = 1
        item.update({'hit':hit})
        results.append(item)
    
    acc = correct/len(task1)
    print(f"* Accuracy for task1: {acc}")
    print("** Number of fail: ", fail)
    
    return acc, results

def eval_task2(results:Union[str, dict]):
    """
    Args:
        result_file (str): Path to the pickle file containing the prediction results

    Returns:
        Tuple of:
        - (Accuracy, F1 score, IoU score)
        - List of updated results with predictions and scores.
    """
    if isinstance(results, str):
        results = pickle.load(open(results,"rb"))
    task2 = results['Task2']

    flavordb = pickle.load(open(Config.db_address,"rb"))
    mid2molecule = flavordb['id2molecule']

    lm = dspy.OpenAI(model='gpt-3.5-turbo', api_key=Config.OPENAI_API_KEY)
    dspy.settings.configure(lm=lm)
    predict_groups = dspy.Predict(predict_functional_groups)

    functional_groups = [ # 53
        'thiocarboxylic',
        'cation',
        'sulfone',
        'hydroxy',
        'sulfonic',
        'alcohol',
        'ketone',
        'hydroxyhetarene',
        'amine',
        'aryl',
        'trialkylamine',
        'carboxylic',
        'alkyne',
        'ketene',
        'anhydride',
        'acetal',
        'amide',
        'derivative',
        'carbonitrile',
        'heterocyclic',
        '(alkylamine)',
        'aliphatic/aromatic',
        'imide,',
        'enol',
        'halide',
        'phenol',
        'sulfoxide',
        'aldehyde',
        'thioether',
        'hydroperoxide',
        'ester',
        'isothiocyanate',
        'alpha-aminoacid',
        'dialkylamine',
        'thiol',
        'ammonium',
        'aliphatic',
        'arylthiol',
        'aromatic',
        'thioacetal',
        'alpha-hydroxyacid',
        'acid',
        'sulfanyl',
        'alkylthiol',
        'salt',
        'alkene',
        'ether',
        'sulfenic',
        'carbonyl',
        'nitrite',
        'halogen',
        'chloride',
        'oxo(het)arene']
    
    def convert_string_to_list(input_str):
        # Remove the surrounding brackets
        cleaned_str = input_str.strip("[]")
        # Split the string by ', ' to get individual elements as strings
        if "'" in cleaned_str:
            list_elements = cleaned_str.split("', '")
            list_elements = [element.strip("'") for element in list_elements]
        elif '"' in cleaned_str:
            list_elements = cleaned_str.split('", "')
            list_elements = [element.strip('"') for element in list_elements]
        else:
            list_elements = cleaned_str.split(",")
            list_elements = [element.strip() for element in list_elements]
        
        return list_elements

    mname2groups = {}
    for i in mid2molecule:
        groups = mid2molecule[i]['functional_groups'].split(" ")
        tmp = []
        for g in groups:
            if "@" in g:
                if "compound" in g:
                    tmp.append(g.split("@")[1])
                if "primary" in g:
                    tmp.append(g.split("@")[0])
            else:
                tmp.append(g)
        groups = tmp
        mname2groups[mid2molecule[i]['common_name']] = groups

    correct = 0
    results = []
    f1s, ious = [], []

    for item in tqdm(task2): 
        gt_molecules = item['actual_molecules'] # masked molecules
        tmp = []
        for m in gt_molecules:
            if m in mname2groups:
                tmp+=mname2groups[m]
        gt_groups = set(tmp)

        try:
            predicted_molecules = ast.literal_eval(item['predicted_molecules'])
            assert isinstance(predicted_molecules, list)
        except:
            predicted_molecules = convert_string_to_list(item['predicted_molecules'])
        
        all_predicted_cates = []
        for m in predicted_molecules: # TODO Predicted Functional Groups:
            try:
                predicted_cates = predict_groups(molecule=m, functional_groups=str(functional_groups)).predicted_functional_groups
                if 'predicted' in predicted_cates.lower():
                    predicted_cates = predicted_cates.split(":")[-1].strip()
                all_predicted_cates += convert_string_to_list(predicted_cates)
            except:
                continue
        
        print("\nGT group: ", gt_groups)
        print("Predicted group: ", set(all_predicted_cates))

        # Calculate the overlap between all_predicted_cates and gt_groups
        all_predicted_cates = set(all_predicted_cates)
        overlap = len(all_predicted_cates.intersection(gt_groups))
        # results.append(list(all_predicted_cates))
        try:
            correct += overlap/len(gt_groups)
        except:
            print(item)
            continue
       
        # IoU, F1 score
        union = len(all_predicted_cates.union(gt_groups))
        iou = overlap / union
        
        precision = overlap / (len(all_predicted_cates)+1e-09)
        recall = overlap / len(gt_groups)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        
        print("IoU: ", iou)
        print("F1: ", f1)

        ious.append(iou)
        f1s.append(f1)

        item.update({'predicted_categories': list(all_predicted_cates),'f1': f1, 'iou': iou})
        results.append(item)

    acc = correct/len(task2) # TODO
    avg_f1 = sum(f1s)/len(f1s) ## sample average ! instead of micro/macro average
    avg_iou = sum(ious)/len(ious)

    print(f"* Accuracy for task2: {acc}")
    print(f"* F1 (sample average): {avg_f1}, IoU (sample average): {avg_iou}")
    
    return (acc, avg_f1, avg_iou), results


if __name__ == '__main__':
    _, result_task1 = eval_task1()
    _, result_task2 = eval_task2()

    eval_result_file='eval_baseline_results_100-gpt-3.5.p'
    pickle.dump({'Task1': result_task1, 'Task2': result_task2}, open(eval_result_file, 'wb'))
    print(f"* Eval results have been dumped to {eval_result_file}")
    print("DONE")
