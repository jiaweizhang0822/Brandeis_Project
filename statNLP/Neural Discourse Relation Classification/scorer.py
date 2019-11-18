"""Please leave this script untouched.

To use:
    python scorer.py [gold_relations.json] [pred_relations.json]
"""
import json
import sys


def accuracy(gold_list, auto_list):
    gold_sense_list = [relation['Sense'] for relation in gold_list]
    auto_sense_list = [relation['Sense'][0] for relation in auto_list]

    correct = len([1 for i in range(len(gold_list))
        if auto_sense_list[i] in gold_sense_list[i]])

    print('Accuracy: {:<13.5}'.format(correct / len(gold_list)), end='\n\n')

def prf_for_one_tag(gold_list, auto_list, tag):
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(gold_list)):
        if tag in gold_list[i]['Sense'] and auto_list[i]['Sense'][0] == tag:
            tp += 1
        elif tag in gold_list[i]['Sense']:
            fn += 1
        elif auto_list[i]['Sense'][0] == tag:
            fp += 1

    p = tp / (tp + fp) if tp + fp != 0 else 0.
    r = tp / (tp + fn) if tp + fn != 0 else 0.
    f = 2 * p * r / (p + r) if p + r != 0 else '-'
    print('{:35} precision {:<13.5}recall {:<13.5}F1 {:<13.5}'.format(tag, p, r, f))

    return tp, fp, fn, p, r, f

def prf(gold_list, auto_list):
    tag_dict = {sense:None for relation in gold_list for sense in relation['Sense']}

    total_tp, total_fp, total_fn, total_p, total_r, total_f = 0, 0, 0, 0, 0, 0
    for tag in tag_dict: 
        tp, fp, fn, p, r, f = prf_for_one_tag(gold_list, auto_list, tag)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_p += p
        total_r += r
        total_f += f if f != '-' else 0

    print()
    p = total_tp / (total_tp + total_fp) if total_tp + total_fp != 0 else 0.
    r = total_tp / (total_tp + total_fn) if total_tp + total_fn != 0 else 0.
    f = 2 * p * r / (p + r) if p + r != 0 else '-'
    print('{:35} precision {:<13.5}recall {:<13.5}F1 {:<13.5}'.format('Micro-Average', p, r, f))
    print()

if __name__ == '__main__':
    gold = sys.argv[1]
    auto = sys.argv[2]

    gold_list = [json.loads(x) for x in open(gold)]
    auto_list = [json.loads(x) for x in open(auto)]

    print('='*60 + '\nEvaluation for all discourse relations\n')
    accuracy(gold_list, auto_list)
    prf(gold_list, auto_list)

    print('='*60 + '\nEvaluation for explicit discourse relations only\n')
    accuracy([g for g in gold_list if g['Type'] == 'Explicit'],
        [a for a in auto_list if a['Type'] == 'Explicit'])
    prf([g for g in gold_list if g['Type'] == 'Explicit'],
        [a for a in auto_list if a['Type'] == 'Explicit'])

    print('='*60 + '\nEvaluation for non-explicit discourse relations only\n')
    accuracy([g for g in gold_list if g['Type'] != 'Explicit'],
        [a for a in auto_list if a['Type'] != 'Explicit'])
    prf([g for g in gold_list if g['Type'] != 'Explicit'],
        [a for a in auto_list if a['Type'] != 'Explicit'])
