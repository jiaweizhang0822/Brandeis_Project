import sys


def compute_acc(gold_file, auto_file):
    gold_lines = open(gold_file).readlines()
    auto_lines = open(auto_file).readlines()

    gold_lines = [[tup.split('_') for tup in line.split()] for line in gold_lines]
    auto_lines = [[tup.split('_') for tup in line.split()] for line in auto_lines]

    correct = 0.0
    total = 0.0
    for g_snt, a_snt in zip(gold_lines, auto_lines):
        correct += sum([g_tup[1] == a_tup[1] for g_tup, a_tup in zip(g_snt, a_snt)])
        total += len(g_snt)

    return correct / total


if __name__ == '__main__':
    gold_file = sys.argv[1]
    auto_file = sys.argv[2]

    print('POS tagging accurarcy: {:.4f}'.format(compute_acc(gold_file, auto_file)))