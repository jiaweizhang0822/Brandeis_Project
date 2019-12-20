import processor as p
import sys

def export_snt(data, out_path):
    str = ""
    for d in data:
        str+= ' '.join(d)+'\n'
    with open(out_path, 'w') as f:
        f.write(str)

def export_word(data, out_path):
    str = ""
    for d in data:
        str += d + '\n'
    with open(out_path, 'w') as f:
        f.write(str)

####################### Output postprocess file #######################
data_dir = './data'
output_dir = './data/postprocess/'
data = p.load_data(data_dir)

model = sys.argv[1]
model_list = [model]

for model in model_list:
    with open('./nmt/tmp/' + model + '/output_test', 'r') as f:
        res = [line.rstrip('\n') for line in f]
    res = [p.postprocess(res[i], data['test'][0][i]) for i in range(len(res))]
    export_word(res, output_dir+'test_'+model)
