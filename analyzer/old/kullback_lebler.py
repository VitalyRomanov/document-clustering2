from numpy import multiply,log

def kld(lm_1,lm_2):
    div = 0.
    for v in lm_1.getVoc():
        div += lm_1.getProb(v)*(log(lm_1.getProb(v))-log(lm_2.getProb(v)))
    return div

def kld_vect(doc1,doc2):
    return multiply(doc1,(log(doc1)-log(doc2))).sum()
