import dspy

def AP_Metric(example, pred, trace=None):
    
    # average precision: https://www.evidentlyai.com/ranking-metrics/mean-average-precision-map
    # compute ap
    original_pred = [i['long_text'] for i in pred['passages']]
    titles = [i.split("|")[0].strip() for i in original_pred]
    answers = {k:1 for k in example.answer}
    seen = 0
    ret = 0
    for i, v in enumerate(titles):
        if v in answers:
            seen += 1
            ret += seen / (i + 1) 
    ret = ret / len(answers)
    return ret


def EM_Metric(example, pred, trace=None):
    original_pred = [i['long_text'] for i in pred['passages']]
    titles = [i.split("|")[0].strip() for i in original_pred]
    shared_elements = set(example.answer) & set(titles)
    ret =  len(shared_elements)/len(set(example.answer))
    return ret

