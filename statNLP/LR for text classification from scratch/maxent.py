# -*- mode: Python; coding: utf-8 -*-
import numpy as np
import scipy
from random import shuffle
from classifier import Classifier
from corpus import ReviewCorpus, BagOfWords, BagOfWordsBigram, BagOfWordsTrigram, Name, NamesCorpus
from nltk.corpus import stopwords
from random import seed
import datetime
import sys

class MaxEnt(Classifier):
    def __init__(self):
        self.models = {'Name':'Logistic Regression'}
        self.vocabulary = {}
        self.weights = None
        self.learning_rate = 0.0001
        self.batcg_size = 1
        self.lambda_l2 = 0
        self.labels = {}
    
    def set_labels(self, train_data):
        labels_set = list(set([i.label for i in train_data]))
        #encoding for each label
        for i in range(len(labels_set)):
            self.labels[labels_set[i]] = i
            
    def get_model(self): 
        return None

    def set_model(self, model): 
        pass

    model = property(get_model, set_model)

    def set_whole_vocabulary(self, train_instances):
        """ get whole vocabulary for train data"""
        n_obs = len(train_instances)
        V = {}
        k = 0
        stopword = stopwords.words('english')
        for i in range(n_obs):
            for j in train_instances[i].features():
                if (j not in V) and (j not in stopword):
                    V[j] = k
                    k+= 1
        self.vocabulary = V
    
    
    def create_feature_mat(self, data):
        '''create feature matrix for every obs in data
           include one extra bias term and one to indicate label
        '''
        num_feature = len(self.vocabulary)
        n_obs = len(data)
        #n_features+1 bias term+1 to 
        feature_mat = np.zeros((n_obs, num_feature+2))
        for i in range(n_obs):
            ith_data_features = data[i].features()
            for j in ith_data_features:
                if j in self.vocabulary:
                    feature_mat[i, self.vocabulary[j]]+= 1
            feature_mat[i, -1] = self.labels[data[i].label]
        feature_mat[:,-2] = 1
        feature_mat = feature_mat.astype(np.int64)
        return feature_mat
    
    def create_feature_vec(self, instance):
        '''create feature vec for a single instance
           include one extra bias term and one to indicate label
        '''
        num_feature = len(self.vocabulary)
        #n_features+1 bias term+1 to 
        feature_vec = np.zeros((num_feature+2,))
        instance_features = instance.features()
        for j in instance_features:
            if j in self.vocabulary:
                feature_vec[self.vocabulary[j]]+= 1
        feature_vec[-1] = self.labels[instance.label]
        feature_vec[-2] = 1
        feature_vec = feature_vec.astype(np.int64)
        return feature_vec
    
    
    def initialize_weights(self):
        '''initialize weights to all zero'''
        k = len(self.labels)
        p = len(self.vocabulary) + 1
        self.weights = np.zeros((k,p))
        

    
    def compute_prob(self, instance):
        '''calculate prob to different categories for a observation (feature vec form)
        '''
        n_cate = self.weights.shape[0]
        prob_vec = np.zeros((n_cate,))
        #e = X @ theta
        e_vec = np.zeros((self.weights.shape[0],))
        #normalized the e score
        for i in range(n_cate):
            e_vec[i] = np.exp(self.weights[i,:]@instance[0:-1]) 
        
        prob_vec = np.exp(e_vec - scipy.special.logsumexp(e_vec))
        
        return prob_vec
    

    def compute_neg_logliklihood(self, minibatch_fea_mat):
        '''
        calculate negative log likelihood for a batch of data 
        That is sum of negative log likelihood (prob of true label) of each observations
        '''
        neg_log = 0
        n_obs = minibatch_fea_mat.shape[0]
        for i in range(n_obs):
            prob_vec = self.compute_prob(minibatch_fea_mat[i,:])
            neg_log+= -np.log(prob_vec[minibatch_fea_mat[i,-1]])
        return neg_log
    
    
    def compute_true_f(self, minibatch_fea_mat):
        '''calculate the gradient conponent for a batch of data: true function  '''
        n_obs = minibatch_fea_mat.shape[0]
        true_f = np.zeros((self.weights.shape))
        for i in range(n_obs):
            true_f[minibatch_fea_mat[i,-1],:]+= minibatch_fea_mat[i,:-1]
        return true_f
    
    def compute_estimated_f(self, minibatch_fea_mat):
        '''calculate the gradient conponent for a batch of data: estimated function  '''
        n_obs = minibatch_fea_mat.shape[0]
        n_cate = len(self.labels)
        est_f = np.zeros((self.weights.shape))
        for i in range(n_obs):
            prob_vec = self.compute_prob(minibatch_fea_mat[i,:])
            for j in range(n_cate):
                est_f[j,:]+= prob_vec[j] * minibatch_fea_mat[i,:-1]
                
        return est_f
    
    def compute_gradient_unreg(self, minibatch_fea_mat):
        '''calculate the gradient conponent for a batch of data: 
            - true function + estimated function without regulization
        '''
        true_f = self.compute_true_f(minibatch_fea_mat)
        est_f = self.compute_estimated_f( minibatch_fea_mat)
        return - true_f + est_f
    
    def compute_gradient_reg(self, minibatch_fea_mat):
        '''calculate the gradient conponent for a batch of data: 
            - true function + estimated function with regulization
        '''
        unreg = self.compute_gradient_unreg(minibatch_fea_mat)
        reg = unreg.copy()
        for i in range(reg.shape[0]):
            reg[i,:]+= reg[i,:] * self.lambda_l2
        return reg
    
    def chop_up(self, train_instances, batch_size):
        '''chop data into minibatches, each size is batch_size'''
        np.random.shuffle(train_instances)
        minibatches = [train_instances[x : x + batch_size] for x in range(0, train_instances.shape[0], batch_size)]
        return minibatches  
    
    def train(self, train_instances, dev_instances=None):
        """Construct a statistical model from labeled instances."""
        self.train_sgd(train_instances, dev_instances, 0.0005, 1000, 0.1, False)

    def train_sgd(self, train_instances, dev_instances, learning_rate, batch_size, lambda_l2, has_v):
        """Train MaxEnt model with Mini-batch Stochastic Gradient 
        """
        #set labels
        self.set_labels(train_instances)
        if not has_v:
            self.set_whole_vocabulary(train_instances)
        train_fea_mat = self.create_feature_mat(train_instances)
        dev_fea_mat = self.create_feature_mat(dev_instances)
        max_iter = 100
        n_iter = 0
        converge = False
        self.initialize_weights()
        self.lambda_l2 = lambda_l2
        curr_neg_loglik = np.inf
        minibatches = self.chop_up(train_fea_mat, batch_size)   
        times = 0
        while (not converge) and (n_iter < max_iter):
            for minibatch in minibatches:
                gradient = self.compute_gradient_reg(minibatch) 
                
                self.weights -= gradient * learning_rate
                prev_neg_loglik = curr_neg_loglik 
                curr_neg_loglik = self.compute_neg_logliklihood(dev_fea_mat)
                print('current negative loglikelihood is ' + str(curr_neg_loglik)
                      + ' at ' + str(n_iter) + ' iteration.')
                print('current accuracy is ' + str(self.accuracy(dev_instances)))
                if (prev_neg_loglik - curr_neg_loglik) / prev_neg_loglik < 0.0005 and (prev_neg_loglik - curr_neg_loglik) / prev_neg_loglik > 0:
                    times +=1
                else:
                    times = 0
                if times == 5:
                    converge = True
                    break
                n_iter+= 1  
                if n_iter > max_iter:
                    break
                
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
    
    def classify(self, instance):
        '''classify a instance (Document object) '''
        instance_fea_vec = self.create_feature_vec(instance)
        prob_vec = self.compute_prob(instance_fea_vec)
        y_pred = np.argmax(prob_vec)
        new_dict = {v : k for k, v in self.labels.items()}
        return new_dict[y_pred]
    
    def accuracy(self, test, verbose=sys.stderr):
        '''calculate accuracy of a corpus object'''
        correct = [self.classify(x) == x.label for x in test]
        return float(sum(correct)) / len(correct)


if __name__ == '__main__':
    start = datetime.datetime.now()
    reviews = ReviewCorpus('yelp_reviews.json', document_class=BagOfWordsBigram)
    seed(hash("reviews"))
    #shuffle(reviews)
    train, dev, test = (reviews[:10000], reviews[10000:11000], reviews[11000:14000])
    end = datetime.datetime.now()
    print (end-start)
    
    
    start = datetime.datetime.now()
    end = datetime.datetime.now()
    print (end-start)
    
    
    start = datetime.datetime.now()
    c1 = MaxEnt()
    c1.set_whole_vocabulary(train)
    V = c1.vocabulary
    c1.set_labels(train)
    c1labels = c1.labels
    end = datetime.datetime.now()
    print (end-start)
    
    start = datetime.datetime.now()
    fea_mat=c1.create_feature_mat(train)
    end = datetime.datetime.now()
    print (end-start)
    
    c1.initialize_weights()
    weight = c1.weights
    c1.train(train,dev)
    minibatches = [fea_mat[x : x + 50] for x in range(0, fea_mat.shape[0], 50)]
    c1.classify(dev[0])
    c1.accuracy(dev)
    dev1 = c1.create_feature_vec(dev[0])
    c1.compute_prob(dev1)
    c1.accuracy(train)
    
    c=0
    for i in range(len(dev)):
        print(c1.classify(dev[i]))
        if  c1.classify(dev[i]) =='positive':
            c+=1
    print(c/len(dev))
        
    names = NamesCorpus(document_class=Name)
    seed(hash("names"))
    shuffle(names)
    #train, dev, test = (names[:5000], names[5000:6000], names[6000:])
    c2 = MaxEnt()
    #c2.train(train, dev)
    #acc = c2.accuracy(test)
    p=0
    for i in range(len(train)):
        if train[i].label =='positive':
            p+= 1
    print(p/len(train))