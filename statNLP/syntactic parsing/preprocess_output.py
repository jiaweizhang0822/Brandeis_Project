import processor as p

def export_snt(data, out_path):
    """
    Args:
        data: list of lists
        out_path:
    """
    str = ""
    for d in data:
        str+= ' '.join(d)+'\n'
    with open(out_path, 'w') as f:
        f.write(str)

def export_word(data, out_path):
    """
    Args:
        data: list of str
        out_path:
    """
    str = ""
    for d in data:
        str += d + '\n'
    with open(out_path, 'w') as f:
        f.write(str)

def inverse(data):
    return

####################### Output preprocess file #######################
data_dir = './data'
output_dir = './data/preprocess/'
data = p.load_data(data_dir)
reverse_data = {}
for key in data.keys():
    reverse_data[key] = [list(reversed(snt)) for snt in data[key][0]]

total_snt = data['train'][0] + data['dev'][0]
vocab_en = set()
for snt in total_snt:
    for word in snt:
        vocab_en.add(word)
vocab_en = list(vocab_en)

linearized = {}
for key in data.keys():
    linearized[key] = [p.linearize_parse_tree(tree) for tree in data[key][1]]

total_parse = linearized['train'] + linearized['dev']
vocab_parse = set()
for snt in total_parse:
    for parse in snt:
        vocab_parse.add(parse)
vocab_parse = list(vocab_parse)
# output vocab, src ends in .en, tgt ends in parse
export_word(vocab_en, output_dir + 'vocab.en')
export_word(vocab_parse, output_dir + 'vocab.parse')

# output train, dev, test file
# src ends in .en, tgt ends in parse
for key in data.keys():
    export_snt(data[key][0], output_dir + key + '.en')
    export_snt(reverse_data[key], output_dir + key + 'reverse.en')
    export_snt(linearized[key], output_dir + key + '.parse')
    export_snt(linearized[key], output_dir + key + 'reverse.parse')



# output gold test
data_dir = './data'
output_dir = './data/preprocess'
data = p.load_data(data_dir)
test_snts = [snt.__str__().replace('\n','') for snt in data['test'][1]]
export_word(test_snts, output_dir+'/gold_output_test')
