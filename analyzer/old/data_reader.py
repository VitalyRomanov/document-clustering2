import MySQLdb
import sys
from getpass import getpass
import os
import pickle



def get_data():
 '''
 Provides the dictionary of articles
 Dictionary key : id of the article as it is stored in the DB
 The article content is stored as a dictionary. The content is
     title       : the title of the article
     body        : the body of the article
     timestamp   : the date of publication
 '''
 passw = getpass(prompt="Password for db: ")
 db = MySQLdb.connect(user      =   "root",
                      passwd    =   passw,
                      db        =   "articles_db",
                      charset   =   'utf8')
 c = db.cursor()

 articles = {}

 id_counter = 0
 c.execute("""SELECT id,title,time,article FROM external_articles""")
 result = c.fetchone()
 while result is not None:
  id_counter += 1
  a_id = result[0]
  a_title = result[1]
  a_time = result[2]
  a_body = result[3]
  articles[id_counter] = {'title':a_title,'body':a_body,'time':a_time}
  result = c.fetchone()
 return articles


def get_all_articles():
    articles_path = "./resources/articles.dat"
    if os.path.isfile(articles_path):
        print("Loading articles from local dump")
        articles = pickle.load(open(articles_path,"rb"))
    else:
        print("Loading articles from db")
        articles = get_data()
        pickle.dump(articles,open(articles_path,"wb"))
    return articles
