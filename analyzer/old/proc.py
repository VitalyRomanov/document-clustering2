from data_reader import get_all_articles
from language_model import MLM,DMLM,dlm_optimal_parameter,Vocabulary
import pickle
from kullback_lebler import kld
import numpy as np
from misc import *


import os
from tokenization import get_document_words

def merge_articles(art):
 c = ""
 for v in art.values():
  c += v['body'] + "\n"
 return c

def prob_dist(lm_1,lm_2):
 dist = 0.
 for w in lm_1.getVoc():
  dist += np.log(lm_2.getProb(w))
 return dist

current_folder = os.getcwd()

create_folders(current_folder)
articles = get_all_articles()


print("Create Collection LM")
collection_text = merge_articles(articles)
voc = Vocabulary(collection_text)
lm_c = MLM(collection_text,voc)
collection_text = "" # free memory

print("Documents' LM")
lm_articles = {}
# for a_id,content in articles.items():
#  lm_articles[a_id] = MLM(content['body'],voc)
# #
# # # with open(lm_folder + "/" + "lm_art.lmc","wb") as lm_art:
# # #  pickle.dump(lm_articles,lm_art)
# #
# print("Calculating optimal parameter")
# mu = dlm_optimal_parameter(lm_articles,lm_c)
# # # mu = 1292.
# # print("Optimal : ",mu)
#
# # print("Appplying smoothing")
# # lm_art_sm = {}
# # for a_id,content in lm_articles.items():
# #  lm_art_sm[a_id] = DMLM(content,lm_c,mu)
#
# # print("Checking distanse")
# # with open("doc_dist_kl.txt","w") as d_dist:
# #  for doc_id,lm_doc_ref in lm_art_sm.items():
# #   d_dist.write("%d "%doc_id)
# #   print("\r%d/%d"%(doc_id,len(lm_art_sm)))
# #   for lm_doc in lm_art_sm.values():
# #    # dist = prob_dist(lm_doc_ref,lm_doc)
# #    dist = kld(lm_doc_ref,lm_doc)
# #    d_dist.write("%f "%dist)
# #   d_dist.write("\n")
#
#
#
# # counter = 0
# # with open(os.getcwd()+"/"+"samples.txt","w") as of:
# #  for a_id,content in articles.items():
# #   counter += 1
# #   of.write("%s\n%s\n\n\n"%(content['title'],content['body']))
# #   print(get_document_words(content['body']))
# #   if counter>2: break
