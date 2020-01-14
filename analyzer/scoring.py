import numpy as np
import sys
class BM25:
    k = 0
    b = 0
    D = 0
    avgln = 0
    _dm = None
    IDF = None

    def IDF_fun(self,nq):
        """
        Calculate log(IDF) for BM25
        """
        N = self._dm.get_n_docs()
        return -np.log(nq/N)

    def __init__(self,dm,k=2.2,b=.75):
        self.k = k
        self.b = b
        self._dm = dm
        v_idf = np.vectorize(self.IDF_fun)
        self.IDF = v_idf(dm.idf)
        # self.IDF = np.array( list( map( self.IDF_fun , dm.idf[0,:] ) ) ).reshape(1,-1)
        self.avgln = np.mean(dm.get_doc_lens())
        self.D = dm.n_docs

    def rank(self, query, score_items=[]):

        # sample documents if available
        if len(score_items)==0:
            wdp = self._dm.wdp
            score_items = np.array(range(self._dm.get_n_docs()))
        else:
            ids = self._dm.get_doc_ids()
            v_find = np.vectorize(lambda x: x in score_items)
            score_items = np.where(v_find(ids))[0]
            wdp = self._dm.wdp[score_items,:]

        # vectorize query
        vq = self._dm.vectorize_query(query).todense()

        # sample words from query
        # second dimension must be len(nzwp)
        nzwp = get_nonzero_words(vq)

        # following two can be used for a LM retrieval
        # vq = filter_words(vq, nzwp)
        # rlm = filter_words(self._dm.rlm, nzwp)

        # filter non-zero words
        idf = filter_words(self.IDF, nzwp)
        wdp = filter_words(wdp, nzwp)

        # filter docs with no overlapping words
        nz_wdm, nz_line_ref = filter_empty_lines(wdp, score_items)

        # apply smoothing using vq and rlm
        dlm = nz_wdm

        # Calculate cost
        # cost = np.zeros((len(nz_line_ref),))
        k = self.k; b = self.b; D = self.D; avgln = self.avgln
        cost = (dlm * (k + 1) * idf / (dlm + k * (1 - b + b * D / avgln))).sum(1)
        ind = np.array(perm_sort(cost))

        ids = self._dm.get_doc_ids()[nz_line_ref]

        return cost[ind], ids[ind]


def get_nonzero_words(bow):
    """
    Returns the positions on nonzero columns in row vector
    """
    return bow.nonzero()[1]


def filter_words(wdm, word_mask):
    """
    Select columns of Wrd-Document Matrix that are specified by word_mask
    """
    return wdm[:, word_mask]


def filter_empty_lines(wdm, line_ref):
    nz_docs = wdm.sum(1).nonzero()[0]
    nz_wdm = wdm[nz_docs, :].toarray()
    nz_line_ref = line_ref[nz_docs]
    return nz_wdm, nz_line_ref


def perm_sort(x):
    return sorted(range(len(x)), key=lambda k: x[k], reverse=True)
