from .language_model import Vocabulary
import numpy as np
from .tokenization import RuTok
from scipy.sparse import coo_matrix,dok_matrix,csr_matrix,dia_matrix
import itertools
import sklearn
import joblib as p


class WDM:
    """
    Document-Term Matrix
    interfaces:
    add_doc(doc): doc is a string or list of strings
    _construct_idf() : calculate idf values
    _construct_rlm() : creates corpus language model of type MLM
    _construct_wdp() : creates document-term probabilities matrix
    __add_to_wdm()
    _tokenize_documents()
    """
    _voc = None  # vocabulary
    _wdm = None  # word document matrix
    _ste = None  # russian stemmer
    _doc_id = None

    rlm = None  # reference (corpus) language model
    wdp = None  # document-term probabilities
    idf = None

    n_docs = 0
    voc_size = 0

    def __init__(self) -> None:
        """
        Create empty vocabulary, empty Document-Term Matrix, and initialize
        russian stemmer
        """
        self._voc = Vocabulary()
        # changing the sparsity of dak_matrix is more efficient
        # dims of DTM are (n_docs,voc_size)
        self._wdm = dok_matrix(np.zeros((0, self._voc.size)))
        self._wdm = self._wdm.tocsr()
        self._ste = RuTok()
        self._doc_id = []

    def get_last_doc_id(self):
        if len(self._doc_id) == 0:
            return -1
        else:
            return self._doc_id[-1]

    def get_doc_ids(self):
        return np.array(self._doc_id)

    def add_docs(self, docs_d):
        """
        Adds document into the Document-Term Matrix (DTM)
        docs : string or list of strings that represent document content
        """
        docs_ids = docs_d['ids']
        docs = docs_d['docs']

        if isinstance(docs, str):
            docs = [docs]
        else:
            print("Adding %d docs to DTM" % len(docs))
            if len(docs) == 0:
                return

        self._doc_id.extend(docs_ids)

        # perform lemmatization of russian words
        docs_tok = self._tokenize_documents(docs)

        # add new words to vocabulary
        all_new_tokens = list(itertools.chain.from_iterable(docs_tok))
        v_s_old, v_s_new = self._voc.expand(all_new_tokens, from_tokens=True)

        wdm_temp = self._wdm.todok()
        n_docs = len(docs)
        old_shape = wdm_temp.shape

        # new shape of DTM is determined by the number of new articles and
        # the number of new words
        new_shape = (old_shape[0]+n_docs, old_shape[1]+v_s_new-v_s_old)
        wdm_temp.resize(new_shape)

        for i in range(n_docs):
            self.__add_to_wdm(old_shape[0]+i, docs_tok[i], wdm_temp)
            print("\rAdded %d out of %d" % (i, n_docs), end="")
        print("\r", end="")

        self._wdm = wdm_temp.tocsr()

        self._construct_wdp()
        self._construct_rlm()
        self._construct_idf()

    def __add_to_wdm(self, doc_id, doc_tokens, wdm):
        """
        add doc to DTM
        doc_id : the row where the document is added
        doc : document tokens
        wdm : DTM
        """
        t_loc = self._voc.find(doc_tokens)
        for t_l in t_loc:
            wdm[doc_id, t_l] += 1

    def _construct_rlm(self):
        """
        Reference Language Model is overall probabilities for
        each word in the corpus
        """
        rlm = self._wdm.sum(0)  # sum word counts over all documents
        self.rlm = sklearn.preprocessing.normalize(rlm, axis=1)  # /=np.sum(rlm) # normalize to overall word count

    def _construct_wdp(self):
        """
        Construct Language Model for each document : the array of probabilities
        of each word for particular document
        """
        # dcs = np.asarray(self._wdm.sum(1)).squeeze(1) # document lengths
        # idcs = 1/dcs
        # idcs[idcs == np.inf] = 0
        # idcs_m = dia_matrix((idcs,0),(len(idcs),len(idcs)))
        #
        # wdp = self._wdm.copy()
        # wdp = idcs_m * wdp
        # return wdp
        self.wdp = sklearn.preprocessing.normalize(self._wdm, axis=1)

    def _construct_idf(self):
        """
        Calculate IDF count for words in vocabulary
        """

        self.idf = (self._wdm > 0).toarray().sum(0, keepdims=True)
        # print(self._voc.get_word([0]),self.idf[0,0])
        # return
        # return np.asarray((self._wdm>0).sum(0)).reshape(-1)

    def _tokenize_documents(self, docs):
        """
        docs : the list of strings that represent documents
        """
        doc_st = []
        for d_id, doc in enumerate(docs):
            doc_st.append(self._ste.tokenize(doc))
            print("\rTokenized %d/%d" % (d_id, len(docs)), end="")
        return doc_st

    # def query_score(self,query,id2):
    #     """
    #     evaluate a document against a given query
    #     query : query vector of dims (voc_size,)
    #     id2 : the id (row) of the document to compare the query with
    #     """
    #     l = 0.1
    #     doc1_w = get_nonzero_words(query) # retrieve nonzero entries in query vector
    #     doc1_raw_lm = sample_raw_doc_lm(query,doc1_w) # sample nonzero query terms from query
    #     doc2_raw_lm = sample_raw_doc_lm(self._wdm[id2,:],doc1_w) # sample nonzero query terms from document
    #     rlm = sample_ref_lm(self._rlm,doc1_w) # sample nonzero query terms from reference lang model
    #     doc1_lm = doc1_raw_lm#doc1_raw_lm*(1-l)+rlm*l # apply Jilinec-Mercer smoothing to query
    #     doc2_lm = doc2_raw_lm*(1-l)+rlm*l # apply Jilinec-Mercer smoothing to document
    #
    #     diff = kld_vect(doc1_lm,doc2_lm)
    #     return diff
    #
    # def similarity_score(self,id1,id2):
    #     l = 0.9
    #     doc1_w = get_nonzero_words(self._wdm[id1,:])
    #     doc1_raw_lm = sample_raw_doc_lm(self._wdm[id1,:],doc1_w)
    #     doc2_raw_lm = sample_raw_doc_lm(self._wdm[id2,:],doc1_w)
    #     rlm = sample_ref_lm(self._rlm,doc1_w)
    #     doc1_lm = doc1_raw_lm*(1-l)+rlm*l
    #     doc2_lm = doc2_raw_lm*(1-l)+rlm*l
    #
    #     diff = kld_vect(doc1_lm,doc2_lm)
    #     # diff = self.sess.run([self.tf_kld],{self.doc1:self._wdm[id1,:].todense(),
    #     #                                     self.doc2:self._wdm[id2,:].todense(),
    #     #                                     self.ref:rlm,
    #     #                                     self.l:l})
    #
    #     return diff

    # def get_vector(self,id):
    #     return self._wdm[id,:].todense()

    # def build_graph(self):
    #     self.doc1 = tf.placeholder(shape=(1,self._voc.size),dtype=tf.float32)
    #     self.doc2 = tf.placeholder(shape=(1,self._voc.size),dtype=tf.float32)
    #     self.ref = tf.placeholder(shape=(1,self._voc.size),dtype=tf.float32)
    #     self.l = tf.placeholder(dtype=tf.float32)
    #
    #     doc1_lm = self.doc1/tf.reduce_sum(self.doc1)*(1-self.l) + self.ref*self.l + 1e-308
    #     doc2_lm = self.doc2/tf.reduce_sum(self.doc2)*(1-self.l) + self.ref*self.l + 1e-308
    #
    #
    #     self.tf_kld = tf.reduce_sum(tf.multiply(doc1_lm,(tf.log(doc1_lm)-tf.log(doc2_lm))))
    #     self.sess = tf.Session()

    def vectorize_query(self, query):
        doc_tokens = self._tokenize_documents([query])[0]
        print("\r                   ")
        # doc_tokens = docs_tok.split(" ")
        t_loc = self._voc.find(doc_tokens)
        query_vect = dok_matrix(np.zeros((1, self._voc.size)))
        for t in t_loc:
            query_vect[0, t] += 1
        return query_vect

    def get_doc_lens(self):
        return np.asarray(self._wdm.sum(1)).reshape(-1)

    def get_n_docs(self):
        return self._wdm.shape[0]

    def get_voc_size(self):
        return self._wdm.shape[1]

    def load(path):
        return p.load(open(path, "rb"))

    def save(self,path):
        p.dump(self, open(path, "wb"), protocol=0)


# def sample_raw_doc_lm(doc_bow, w_ids):
#     """
#     doc_bow : sparse array with dims [1,voc_size]
#     w_ids : array of positions of words of interest
#     """
#     doc_raw_lm = doc_bow[:,w_ids].todense().reshape(-1,1).squeeze(1)
#     doc_raw_lm /= np.sum(doc_bow)
#     return doc_raw_lm
#
# def sample_ref_lm(ref_lm,w_ids):
#     return ref_lm[:,w_ids].reshape(-1,1).squeeze(1)
#
# def get_nonzero_words(doc_raw_lm):
#     return doc_raw_lm.nonzero()[1]
