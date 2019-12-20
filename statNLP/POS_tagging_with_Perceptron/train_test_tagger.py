import sys
from perceptron_pos_tagger import Perceptron_POS_Tagger
from data_structures import Corpus, Sentence
import datetime

def read_in_gold_data(filename):
    '''
    read data with tags
    '''
    with open(filename) as f:
        lines = f.readlines()
        lines = [[tup.split('_') for tup in line.split()] for line in lines]
        sents = [Sentence(line,gold=True) for line in lines]

    return sents 


def read_in_plain_data(filename):
    '''
    read data without tags
    '''
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        sents = [Sentence(line,gold=False) for line in lines]

    return sents 


def output_auto_data(tagger, auto_data, filename):
    '''  output your auto tagged data into a file,
        format: provided gold data (i.e. word_pos word_pos ...). 
    '''
    f = open(filename, 'w')
    j=0
    for data in auto_data:
        print('output',j)
        j+=1
        route = tagger.tag(data)
        route = route[1: -1]
        s = ''
        for i in range(len(route)):
            s += (str(data.origin_snt[i]) + '_' + str(route[i]) + ' ')
        f.write(s)
        f.write('\n')
    f.close()


def read_raw_data(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [[tup.split('_') for tup in line.split()] for line in lines]
    return lines
        
    
if __name__ == '__main__':

    # Run python train_test_tagger.py train/ptb_02-21.tagged dev/ptb_22.tagged dev/ptb_22.snt test/ptb_23.snt to train & test your tagger
    #train_file = sys.argv[1]
    #gold_dev_file = sys.argv[2]
    #plain_dev_file = sys.argv[3]
    #test_file = sys.argv[4]

    train_file = 'train/ptb_02-21.tagged'
    gold_dev_file = 'dev/ptb_22.tagged'
    plain_dev_file = 'dev/ptb_22.snt'
    test_file = 'test/ptb_23.snt'
    
    # Read in data
    train_data = read_in_gold_data(train_file)
    gold_dev_data = read_in_gold_data(gold_dev_file)
    plain_dev_data = read_in_plain_data(plain_dev_file)
    test_data = read_in_plain_data(test_file)
    
    # Train your tagger
    a = Corpus('train/ptb_02-21.tagged','dev/ptb_22.tagged')
    
    my_tagger = Perceptron_POS_Tagger(a.tags)
    start = datetime.datetime.now()
    #my_tagger.train(train_data, gold_dev_data)
    my_tagger.train_avg_perceptron(train_data[0:5000], gold_dev_data)
    end = datetime.datetime.now()
    #my_tagger.compute_acc(gold_dev_data)
    print('training takes', end - start)
    
    output_auto_data(my_tagger, plain_dev_data, 'dev/pred_ap.tagged')
    output_auto_data(my_tagger, test_data, 'test/pred_ap.tagged')

    
 


