import json
import os
def string_iou(str1, str2, mode='word'):
    """
    Compute Intersection over Union (IoU) between two strings.

    Args:
        str1 (str): First string
        str2 (str): Second string
        mode (str): 'word' for word-level, 'char' for character-level comparison

    Returns:
        float: IoU score between 0 and 1
    """
    if mode == 'word':
        set1 = set(str1.lower().split())
        set2 = set(str2.lower().split())
    elif mode == 'char':
        set1 = set(str1.lower())
        set2 = set(str2.lower())
    else:
        raise ValueError("mode must be 'word' or 'char'")

    intersection = set1 & set2
    union = set1 | set2

    if not union:
        return 0.0

    return len(intersection) / len(union)


def get_score(true_triplet : dict, predicted_triplet : dict, threshold : float):
    tsubj, tobj = true_triplet["subject"].lower(), true_triplet["object"].lower()
    psubj, pobj = predicted_triplet["subject"].lower(), predicted_triplet["object"].lower()
    subject_iou =  string_iou(tsubj, psubj)
    if subject_iou < threshold:
        return 0
    object_iou =  string_iou(tobj, pobj)
    if object_iou < threshold:
        return 0
    return (subject_iou+object_iou)/2

def get_predicitons(true_triplets : list, predicted_triplets : list, threshold : float):
    scores = {}
    for i, true_triplet in enumerate(true_triplets):
        for j, predicted_triplet in enumerate(predicted_triplets):
            id = f"{i}_{j}"
            if id not in scores:
                scores[id] = get_score(true_triplet, predicted_triplet, threshold)
    sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    tp,already_true_map, already_pred_map = 0, {}, {}
    for id, score in sorted_scores.items():
        id = id.split('_')
        true_id = id[-0]
        pred_id = id[-1]
        if true_id not in already_true_map and pred_id not in already_pred_map:
            already_true_map[true_id] = True
            already_pred_map[pred_id] = True
            if score > 0:
                tp += 1 

    return tp/len(true_triplets) if len(true_triplets) > 0 else 0



def compute_evaluation(name : str, property_types : list, true_triplets : list, predicted_triplets : list, threshold : float = 0.8):
    evaluation = {}
    mean_accuracy = 0
    for property_type in property_types:
        # get all for this type
        current_true_triplets = [t for t in true_triplets if t['property_type'] == property_type]
        if len(current_true_triplets) == 0:
            continue
        current_predicted_triplets = [t for t in predicted_triplets if t['property_type'] == property_type]
        acc = get_predicitons(current_true_triplets, current_predicted_triplets, threshold=threshold)*100
        mean_accuracy += acc 
        evaluation[property_type] = acc
    evaluation["mean_accuracy"] = mean_accuracy / len(property_types)

    output_file = open(os.path.join("evaluation", f"{name}_evaluation.json"), "w")
    json.dump(evaluation, output_file)
    output_file.close()
    return evaluation


"""true_triplets = json.load(open("evaluation/true_triplets.json", encoding='utf-8'))
predicted_triplets = json.load(open("evaluation/graph_1.json", encoding='utf-8'))
name = 'graph_1'
threshold = 0.8
property_names = ['participatesIn', 'worksFor', 'hasRole', 'worksOn', 'organizes', 'hasSkill', 'hasDuration', 'locatedIn']

compute_evaluation(name, property_names, true_triplets, predicted_triplets, threshold)"""