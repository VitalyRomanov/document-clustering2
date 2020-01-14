from collections import Counter
from tokenization import get_document_words
import pickle
from pymystem3 import Mystem
import numpy as np
import tensorflow as tf
import sys
from scipy.sparse import dok_matrix,csr_matrix,csc_matrix,lil_matrix


'''
implements several language models that include smoothing:
    MLM : Multinomial language model
    JMLM : Language model with Jelinek Mercer smoothing
    DLM : Language model with Dirichlet smoothing
reference : https://nlp.stanford.edu/~wcmac/papers/20050421-smoothing-tutorial.pdf
'''

LOW_VALUE = sys.float_info.min

# def find_in_voc(voc,tokens):
#     return [voc.index(token) for token in tokens]

class Vocabulary:
    def __init__(self,text):
        self.tokens = list(set(get_document_words(text)))
        self.index = dict(zip(self.tokens,range(len(self.tokens))))
        self.size = len(self.tokens)

    def find(self,tokens):
        return [self.index[token] for token in tokens]

class MLM:
    '''
    Classical language model
    Counts number of word occurences in a diven document
    doc : document text used for generating language model
    '''
    def __init__(self,doc,voc):
        self.voc = voc
        words = get_document_words(doc)
        wc = Counter(words)
        self.tc = sum(wc.values())
        row = np.array(voc.find(wc.keys()))
        col = np.zeros(row.shape)
        val = np.array(list(wc.values()))
        self.wc = csc_matrix((val,(row,col)),shape=(voc.size,1))
        if self.tc!=0:
            self.prob = csc_matrix((val/self.tc,(row,col)),shape=(voc.size,1),dtype=np.float32)
        else:
            self.prob = csc_matrix((voc.size,1),dtype=np.float32)



    def getProb(self,token):
        token_id = self.voc.find([token])[0]
        # returns probability of a token
        if self.tc != 0:
            return self.wc[token_id,0]/self.tc+LOW_VALUE
        else:
            # in case the document is empty
            return LOW_VALUE

    def getCount(self,token):
        token_id = self.voc.find([token])[0]
        return self.wc[token_id,0]

    def getVoc(self):
        return self.wc.nonzero()

    def store(self,name):
        # write the LM on disk
        with open(name,"wb") as lm_file:
            pickle.dump(self,lm_file)


class JMLM:
    '''
    Language model with Jelinek Mercer smoothing
    d_lm : document language model
    r_lm : reference language model, usually LM of the collection
    l : smoothing constant
    '''
    def __init__(self,d_lm,r_lm,l):
        self.d_lm = d_lm
        self.r_lm = r_lm
        self.l = l

    def getProb(self,token):
        p_d = self.d_lm.getProb(token)
        p_r = self.r_lm.getProb(token)
        l = self.l
        return (1-l)*p_d+l*p_r

    def getCount(self,token):
        raise NotImplemented

    def getVoc(self):
        return self.d_lm.lm.keys()

class DMLM:
    '''
    Language model with Dirichlet smoothing
    d_lm : document language model
    r_lm : reference language model, usually LM of the collection
    mu : smoothing constant
    '''
    def __init__(self,d_lm,r_lm,mu):
        self.d_lm = d_lm
        self.r_lm = r_lm
        self.mu = mu

    def getProb(self,token):
        l = mu / (self.d_lm.tc + mu)
        return (1 - l) * self.d_lm.getProb(token) + l * self.r_lm.getProb(token)

    def getCount(self,token):
        raise self.d_lm.getCount(token)

    def getVoc(self):
        return self.d_lm.lm.keys()


def dlm_optimal_parameter(lm_docs,lm_c):
    '''
    Find the optimal parameter for language model with Dirichlet smoothing
    Reference : Two-Stage Language Models for Information Retrieval by
                ChengXiang Zhai and John Lafferty
    lm_docs : dictionaty of language models with doc_id as key
    lm_c : language model of a colection
    '''
    mu = 1200.

    d_size = len(lm_docs)   # number of documents
    v_size = lm_c.voc.size      # vocabulary size
    c_w_ds = lil_matrix((d_size,v_size),dtype=np.float32) # doc. word. freq. mat.
    p_w_cs = lm_c.prob # collection prob
    d_is = np.zeros([d_size,1], dtype=np.float32) # doc length
    for lm_id,lm_d in enumerate(lm_docs.values()):
        print("\r%d/%d"%(lm_id,len(lm_docs)),end='')
        c_w_ds[lm_id,:] = lm_d.wc
        d_is[lm_id,0] = lm_d.tc
    print("")
    c_w_ds = csc_matrix(c_w_ds)


    pickle.dump(c_w_ds,open("doc_freq_matr","wb"))
    pickle.dump(d_is,open("doc_len","wb"))
    pickle.dump(p_w_cs,open("ref_prob","wb"))

    g_mu_w = lambda w_id,mu: np.sum((c_w_ds[:,w_id] * ((d_is - 1) * p_w_cs[0,w_id] - c_w_ds[:,w_id] + 1)) / \
                                (( d_is - 1 + mu ) * ( c_w_ds[:,w_id] - 1 + mu * p_w_cs[0,w_id])))

    g_d_mu_w = lambda w_id,mu: np.sum(- (c_w_ds[:,w_id] * ( (d_is - 1) * p_w_cs[0,w_id] - c_w_ds[:,w_id] + 1)**2) / \
                                    ((d_is - 1 + mu)**2 * (c_w_ds[:,w_id] - 1 + mu * p_w_cs[0,w_id])**2))

    for i in range(1000):

        # g_mu = sum([g_mu_w(w_id,mu) for w_id in range(v_size)])
        # g_d_mu = sum([g_d_mu_w(w_id,mu) for w_id in range(v_size)])

        g_mu = 0.;g_d_mu = 0.
        for w_id in range(v_size):
            g_mu += g_mu_w(w_id,mu)
            g_d_mu += g_d_mu_w(w_id,mu)
            print(g_mu,g_d_mu)

        mu = mu - g_mu / g_mu_d
        print("Iteration %d : %f"%(i,mu))

    # d_is = np.zeros([v_size,1], dtype=np.float32)
    # # p_w_cs = np.zeros([v_size,1], dtype=np.float32)
    # c_w_ds = np.zeros([v_size,1], dtype=np.float32)
    #
    # for i in range(1000):
    #     g_mu = 0.; g_mu_d = 0.
    #     for w_i,w in enumerate(lm_c.lm.keys()):
    #         p_w_cs = lm_c.getProb(w); pos = 0
    #         for lm_doc in lm_docs.values():
    #             c_w_da = lm_doc.lm[w]; d_ia = lm_doc.tc
    #             d_is[pos,0] = d_ia;# p_w_cs[pos,1] = p_w_c;
    #             c_w_ds[pos,0] = c_w_da
    #             pos += 1
    #         g_mu += np.sum((c_w_ds * ((d_is - 1) * p_w_cs - c_w_ds + 1)) / (( d_is - 1 + mu ) * ( c_w_ds - 1 + mu * p_w_cs)))
    #         g_mu_d += np.sum(- (c_w_ds * ( (d_is - 1) * p_w_cs - c_w_ds + 1)**2) / ((d_is - 1 + mu)**2 * (c_w_ds - 1 + mu * p_w_cs)**2))
    #         # print("\r %d/%d   "%(w_i,len(lm_c.lm)), end='')
    #     # print("")
    #     mu = mu - g_mu / g_mu_d
    #     print("Iteration %d : %f"%(i,mu))



    # for i in range(1000):
    #     g_mu = 0.; g_mu_d = 0.
    #     for w_i,w in enumerate(lm_c.lm.keys()):
    #         p_w_c = lm_c.getProb(w)
    #         for lm_doc in lm_docs.values():
    #             c_w_d = lm_doc.lm[w]
    #             d_i = lm_doc.tc
    #             # print((( d_i - 1 + mu ) * ( c_w_d - 1 + mu * p_w_c)))
    #             # print(d_i," ", mu," ", c_w_d," ", p_w_c)
    #             g_mu += (c_w_d * ((d_i - 1) * p_w_c - c_w_d + 1)) / \
    #                     (( d_i - 1 + mu ) * ( c_w_d - 1 + mu * p_w_c))
    #             g_mu_d += - (c_w_d * ( (d_i - 1) * p_w_c - c_w_d + 1)**2) / \
    #                         ((d_i - 1 + mu)**2 * (c_w_d - 1 + mu * p_w_c)**2)
    #         print("\r %d/%d   "%(w_i,len(lm_c.lm)), end='')
    #     print("")
    #     mu = mu - g_mu / g_mu_d
    #     print("Iteration %d : %f"%(i,mu))

    return mu

def mlm_optimal_parameter_tf(lm_docs,lm_c):
    v_size = len(lm_docs)

    with tf.device("/gpu:0"):
        d_is = tf.placeholder(tf.float32,shape=(v_size,1),name="doc_len")
        c_w_ds = tf.placeholder(tf.float32,shape=(v_size,1),name="doc_freq_count")
        p_w_cs = tf.placeholder(tf.float32,name="word_prob")
        # p_w_cs = tf.Variable(0,0,dtype=tf.float32,name="word_prob")

        mu = tf.Variable(1.01,dtype=tf.float32,name="mu")
        g = tf.Variable(0.,dtype=tf.float32,name="g_mu")
        g_d = tf.Variable(0.,dtype=tf.float32,name="g_mu_d")

        g_mu_num = tf.multiply(c_w_ds,((d_is - 1)*p_w_cs - c_w_ds + 1))
        g_mu_den = tf.multiply( d_is - 1 + mu, c_w_ds - 1 + p_w_cs*mu)
        g_mu = tf.reduce_sum(tf.div(g_mu_num,g_mu_den))

        g_mu_d_num = -tf.multiply(c_w_ds,tf.square((d_is - 1)*p_w_cs - c_w_ds + 1))
        g_mu_d_den = tf.multiply( tf.square(d_is - 1 + mu), tf.square(c_w_ds - 1 + p_w_cs*mu))
        g_mu_d = tf.reduce_sum(tf.div(g_mu_d_num,g_mu_d_den))

        g_ass = tf.assign(g,g+g_mu)
        g_d_ass = tf.assign(g_d,g_d+g_mu_d)
        mu_ass = tf.assign(mu,mu-g_ass/g_d_ass)

        init = tf.global_variables_initializer()

        d_i = np.zeros([v_size,1],dtype=np.float32)
        c_w_d = np.zeros([v_size,1],dtype=np.float32)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("./graphs",sess.graph)
        print("Start computation")
        sess.run(init)
        mu_final = 0.
        for i in range(1000):
            for w_i,w in enumerate(lm_c.lm.keys()):
                p_w_c = lm_c.getProb(w); pos = 0
                for lm_doc in lm_docs.values():
                    c_w_da = lm_doc.lm[w]; d_ia = lm_doc.tc
                    d_i[pos,0] = d_ia;# p_w_cs[pos,1] = p_w_c;
                    c_w_d[pos,0] = c_w_da
                    pos += 1
                sess.run([g_ass,g_d_ass],{d_is:d_i,c_w_ds:c_w_d,p_w_cs:p_w_c})
                print("\r %d/%d   "%(w_i,len(lm_c.lm)), end='')
            print("")
            tf.assign(mu,mu-g/g_d)
            mu_final = sess.run(mu_ass)
            print("Iteration %d : "%i,mu_final)
    return mu_final
