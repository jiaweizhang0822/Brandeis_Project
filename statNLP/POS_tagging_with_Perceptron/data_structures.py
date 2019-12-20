import numpy as np
from perceptron_pos_tagger import Perceptron_POS_Tagger


class Corpus(object):
    
    def __init__(self, train_file, dev_file):
        self.train_file = train_file
        self.dev_file = dev_file
        self.tags, self.vocab = self.get_total_features() 
       
    def read_raw_data(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [[tup.split('_') for tup in line.split()] for line in lines]
        
        return lines

    def get_total_features(self):
        '''
        get total feaures and tags
        '''
        train_rawdata = self.read_raw_data(self.train_file)
        dev_rawdata = self.read_raw_data(self.dev_file)
        
        tags = [word[1] for line in train_rawdata for word in line]
        tags = sorted(list(set(tags)))
        tags.extend(['Start', 'End'])
        vocab = [word[0] for line in train_rawdata for word in line] + \
            [word[0] for line in dev_rawdata for word in line]
        vocab = list(set(vocab))
        vocab.extend(['Start1', 'Start2', 'End1', 'End2', 'End3', 'BIAS', 'OOV'])
       
        return tags, vocab


class Sentence(object):
    def __init__(self, snt, gold):
        ''' Modify if necessary
        '''
        #snt = [word.lower() for word in snt]
        self.origin_snt = snt
        if gold:
            self.snt = [['Start1', 'Start']] + [['Start2', 'Start']] + snt + \
                   [['End1', 'End']] + [['End2', 'End']] + [['End3', 'End']]
        else:
            self.snt = ['Start1'] + ['Start2'] + snt + \
                   ['End1'] + ['End2'] + ['End3']
        self.gold = gold
        if gold:          
            self.true_tag = [word[1] for word in snt]
    
    def get_em_features(self, position, tag):
        ''' Implement your feature extraction code here. This takes annotated or unannotated sentence
        and return a set of features
        
        f1: w-2
        f2: w-1
        f3: w0
        f4: w1
        f5: w2
        f6: bias
        position in range (0, origin_snt length)
        '''
        position+= 2
        pos_index = [-2, -1, 0, 1, 2]
        words = self.snt[position - 2 : position + 3]
        
        #pos_index = [-2, -1, 0]
        #words = self.snt[position - 2 : position + 1]
        
        #pos_index = [0, 1, 2]
        #words = self.snt[position : position + 3]
        
        #pos_index=[0]
        #words = [self.snt[position]]
        if self.gold:
            words = [word[0] for word in words]
        else:
            words = [word for word in words]
        features = [(tag, word, i) for word, i in zip(words, pos_index)]
        
        features.append((tag, 'BIAS'))
        features.append((tag, words[2][0:3], "p"))
        features.append((tag, words[2][-3:], "s"))
        return features
    
            
    def get_all_features(self, position, tag, prev_tag):
        ''' Implement your feature extraction code here. This takes annotated or unannotated sentence
        and return a set of features
        
        f1: w-2
        f2: w-1
        f3: w0
        f4: w1
        f5: w2
        f6: bias
        *f7: will be add later (y-1)
        position in range (0, len(snt)-4) or (0, origin_snt length)
        '''
        features = self.get_em_features(position, tag)
        features.append((tag, prev_tag))
        
        return features


if __name__ == '__main__':  
        
    a = Corpus('train/ptb_02-21.tagged','dev/ptb_22.tagged')

    filename = 'train/ptb_02-21.tagged'
    with open(filename) as f:
        train_data = f.readlines()
        train_data = [[tup.split('_') for tup in line.split()] for line in train_data]
        
    data = Sentence(train_data[1])
    data.get_all_features(1,'TO', 'TO')
    data.get_em_features(5,'TO')
#    data.get_total_wts(0,a.wts_dict,'TO','Start')  
    pos = Perceptron_POS_Tagger(a.tags)
    data_tag = pos.tag(data)
    