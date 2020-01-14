# -*- coding: utf-8 -*-

import numpy as np
import pickle as p
from collections import Counter
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize
import fasttext
import gensim

class W2Vhash:
    # voc = None
    # emb = None
    docs = None
    model = None
    dims = 0
    def __init__(self,w2v_path,model_t):
        self.model_t = model_t
        # self.voc = p.load(open(w2v_path + "/voc.pkl","rb"))
        # self.emb = np.load(w2v_path + "/w2v.npy")
        self.index = None
        self.load_model()
        self.docs = np.zeros((0,self.dims))



    def load_model(self):
        if self.model_t=="fasttext":
            model_path = "/Users/LTV/dev_projects/language data/fasttext/fasttext_ru_50000.tar/ru.bin"
            self.model = fasttext.load_model(model_path, encoding='utf-8')
            self.dims = self.model.dims
        elif self.model_t=="w2v":
            model_path = "/Users/LTV/Downloads/all.norm-sz100-w10-cb0-it1-min100.w2v"
            self.model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True, unicode_errors='ignore')
            self.dims = self.model.vector_size
        else:
            raise NotImplemented

    def get_vector(self,word):
        if self.model_t=="fasttext":
            try:
                vect = np.array(self.model[word])
            except:
                vect = np.array([])
        elif self.model_t=="w2v":
            try:
                vect = self.model.word_vec(word)
            except:
                vect = np.array([])
        else:
            raise NotImplemented

        return vect



    def add_docs(self,docs,tokenized=True):
        if not tokenized:
            raise NotImplemented
        docs_tf = []

        new_docs = np.zeros((len(docs),self.dims))
        doc_emb = np.zeros((1,self.dims))

        for doc_id,doc in enumerate(docs):
            doc_tf = Counter(doc)
            docs_tf.append(doc_tf)

        for doc_id,doc_tf in enumerate(docs_tf):
            term_w = []

            term_mat = np.zeros((len(doc_tf),self.dims))
            t_id = 0
            for term in doc_tf:
                w_vec = self.get_vector(term)
                if len(w_vec)==0: continue
                term_mat[t_id,:] = w_vec
                term_w.append(doc_tf[term])
                t_id += 1

            term_weights = np.array(term_w)/sum(term_w)
            # doc_emb = np.average(term_mat[:t_id,:],weights=term_weights,axis=0)
            doc_emb = term_mat[:t_id,:].max(0)
            new_docs[doc_id,:] = doc_emb

        self.docs = np.append(self.docs,new_docs,axis=0)
        self.docs = normalize(self.docs,axis=1)

        self.index = KDTree(self.docs, leaf_size=30, metric='euclidean')





if __name__=="__main__":

    def load_articles():
        articles_dump = "../res/dataset.dat"
        if not os.path.isfile(articles_dump):
            dataset = AData()
            dataset.load_new()
            p.dump(dataset,open(articles_dump,"wb"))
        dataset = p.load(open(articles_dump,"rb"))
        print("Article dump loaded")
        return dataset

    from tokenization import RuTok
    import os
    tok = RuTok()
    dataset = load_articles()

    a_tok = []

    articles = dataset.titles[:1000]

    for a_id,article in enumerate(articles):
        a_tok.append(tok.tokenize(article))
        print("\rTokenizing %d/%d"%(a_id+1,len(articles)),end="")

    w2vhash = W2Vhash("../res",'w2v')
    w2vhash.add_docs(a_tok)

    # p.dump(w2vhash,open("w2vhash.pkl","wb"))

    for query_doc_id  in range(100):
        dist,ind = w2vhash.index.query(w2vhash.docs[query_doc_id,:].reshape(1, -1), k=20, return_distance=True)

        print("\n",query_doc_id,dataset.titles[query_doc_id])
        for ii,i in enumerate(ind[0]):
            print(dist[0][ii],dataset.titles[i])
