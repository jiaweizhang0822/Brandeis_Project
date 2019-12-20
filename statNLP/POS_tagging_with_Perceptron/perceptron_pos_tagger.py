import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from numba import jit

class Perceptron_POS_Tagger(object):
    def __init__(self, tags):
        ''' Modify if necessary. 
        '''
        self.tags = tags
        self.wts = defaultdict(int)
        for tag in tags:
            self.wts[(tag, 'BIAS')] = 1
    

    def sum_features(self, features): 
        ''' get sum of all features'''
        
        return np.sum([self.wts[feature] for feature in features])
    

    def tag(self, data):
        '''
        viterbi algorithm to tag the sequence
        '''
         #origin sentence and 1 end 
        col = len(data.origin_snt)
        tags = self.tags
        row = len(tags)
        #score_mat to store the max score from begin to a current node
        score_mat = np.zeros((row, col))
        features_tags = [data.get_all_features(0, tag, 'Start') for tag in tags]
        score_mat[:,0] = [self.sum_features(feature_tag) for feature_tag in features_tags]

        #to store the route, that is every tags's prev tag
        route = pd.DataFrame('', index = tags, columns = range(col))

        for i in range(1, col):             
            for j in range(row):
                curr_tag = tags[j]
                em_features = data.get_em_features(i, curr_tag) 
                emission_score = self.sum_features(em_features)
                transition_score = [self.wts[(curr_tag, tag)] for tag in tags]
                compare_vec = score_mat[:, i - 1] + transition_score
                prev_tag = tags[np.argmax(compare_vec)]
                route.loc[curr_tag, i] = prev_tag
                score_mat[j, i] = np.max(compare_vec) + emission_score
        
        em_features = data.get_em_features(col, 'End') 
        emission_score = self.sum_features(em_features)
        transition_score = [self.wts[('End', tag)] for tag in tags]
        compare_vec = score_mat[:, col - 1] + transition_score
        
        end_total = compare_vec + emission_score
        curr_tag = 'End' 
        data_tag = [curr_tag]
        prev_tag = tags[np.argmax(end_total)]
        #prev_tag = tags[np.argmax(score_mat[:,col - 1])]
        data_tag.append(prev_tag)
        
        
        for i in range(col - 1, 0, -1):   
            curr_tag = prev_tag
            prev_tag = route.loc[curr_tag, i]
            data_tag.append(prev_tag)
            
        data_tag.reverse()

        return ['Start']+ data_tag
    
    
    def update_wts(self, features_F, features_T):
        '''
        update the wts
        increase wts of true features by 1
        decrease wts of wrong features by 1
        '''
        for feature in features_F:
            self.wts[feature]-= 1
        for feature in features_T:
            self.wts[feature]+= 1


    
    def train(self, train_data, dev_data):
        ''' Implement the Perceptron training algorithm here.
        train_data and dev_data are both list of Sentence object
        
        '''
        for iter_ in range(3):
            for i in range(len(train_data)):
                print('deal instance', i)
                data = train_data[i]
                pred_tag = self.tag(data)
                pred_tag = pred_tag[:-1]
                true_tag = ['Start'] + data.true_tag
                ln_snt = len(data.origin_snt)
                for j in range(1, ln_snt + 1):
                    if pred_tag[j] != true_tag[j]:
                        features_F = data.get_em_features(j-1, pred_tag[j])
                        features_T = data.get_em_features(j-1, true_tag[j])
                        self.update_wts(features_F, features_T)
                    self.wts[(true_tag[j], true_tag[j-1])]+= 1
                    self.wts[(pred_tag[j], pred_tag[j-1])]-= 1
                
                self.wts[('End', true_tag[-1])]+= 1
                self.wts[('End', pred_tag[-1])]-= 1
            #acc = self.compute_acc(dev_data[0:10])
            #print('dev acc', acc)
        
    
    def train_avg_perceptron(self, train_data, dev_data):
        '''
        average perceptron algorithm
        
        '''
        avg_wts = defaultdict(int)
        for i in range(len(train_data)):
            print('deal instance', i)
            data = train_data[i]
            pred_tag = self.tag(data)
            pred_tag = pred_tag[:-1]
            true_tag = ['Start'] + data.true_tag
            ln_snt = len(data.origin_snt)
            for j in range(1, ln_snt + 1):
                if pred_tag[j] != true_tag[j]:
                    features_F = data.get_em_features(j-1, pred_tag[j])
                    features_T = data.get_em_features(j-1, true_tag[j])
                    self.update_wts(features_F, features_T)
                self.wts[(true_tag[j], true_tag[j-1])]+= 1
                self.wts[(pred_tag[j], pred_tag[j-1])]-= 1
            
            self.wts[('End', true_tag[-1])]+= 1
            self.wts[('End', pred_tag[-1])]-= 1

            for k in self.wts.keys():
                avg_wts[k]+= self.wts[k]
        self.wts = avg_wts
        
    
    def compute_acc(self,dev_data):
        '''
        compute accuracy of dev data
        '''
        correct = 0.0
        total = 0.0
        for i in range(len(dev_data)):
            pred_tags = self.tag(dev_data[i])
            pred_tags = pred_tags [1:-1]
            gold_tags = [data[1] for data in dev_data[i].origin_snt]
            correct += sum(p_tag==g_tag for p_tag, g_tag in zip(pred_tags, gold_tags))
            total += len(gold_tags)
            print('dev',i)
            print(correct, total)
    
        return correct / total
    
    def output_dev(self, dev_data):
        '''
        output prediction of dev/test data
        '''
        f = open('dev/pred.tagged', 'w')
        for data in dev_data:
            route = self.tag(data)
            route = route[1: -1]
            s = ''
            for i in range(len(route)):
                s += (str(data.origin_snt[i][0]) + '_' + str(route[i]) + ' ')
            f.write(s)
            f.write('\n')
        f.close()