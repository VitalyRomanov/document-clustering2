# import os
# import joblib as p
from .article_loader import AData, post_json
import json
from .WDM import WDM
import os
from .scoring import BM25
import sys
from .http_server import HTTPServer, nddHandler, HOST_NAME, PORT_NUMBER
import time
from bs4 import BeautifulSoup
import urllib.request
import json

def post_text(text):
    url_addr = "http://localhost:8081/v2/check"
    data = urllib.parse.urlencode({'language':'ru', 'text':text}).encode('utf-8')
    req = urllib.request.Request(url_addr, data=data, headers={'content-type': 'application/json'})
    response = urllib.request.urlopen(req)

    res = json.loads(response.read().decode('utf-8'))

    return [text[err['offset']:err['offset']+err['length']] for err in res['matches']]

class Analyzer():
    dataset = None
    dtm_index = None
    search = None
    params = None

    def __init__(self,params):
        self.params = params

    def begin(self):
        # unpack parameters
        index_by = self.params['index_by']
        enable_filtering = self.params['enable_filtering']
        res_path = self.params['resources']
        # articles_dump = params['articles_dump']
        # dtm_dump = params['dtm_dump']
        similarity_threshold = self.params['similarity_threshold']
        server_mode = self.params['server_mode']

        print("\n\nSpinning up...")
        # load data
        self.dataset = load_articles(res_path)
        # print(len(dataset.content), "\n\n\n")

        # f_ind = -1
        # c_file = open("content_%d.txt"%f_ind,"w")
        # for i,a in enumerate(self.dataset.content):
        #     f_ind += 1
        #     if f_ind % 1000 == 0:
        #         c_file.close()
        #         c_file = open("content_%d.txt"%f_ind,"w")
        #     soup = BeautifulSoup(a,'html.parser')
        #     text = soup.get_text()
        #     c_file.write("%s\n" % "\n".join(post_text(text)))
        #     # sys.exit()
        # c_file.close()

        # load index
        self.dm = load_dtf(res_path, self.dataset, {'index_by': index_by, 'enable_filtering': enable_filtering})

        if server_mode:
            print("\nSwitching in server mode")
            # init search engine
            print("Initializing search engine...", end="")
            search_bm25 = BM25(self.dm)
            print("done")
            launch_server(self.dataset, search_bm25, HOST_NAME, PORT_NUMBER, res_path)

        else:
            print("\nSwitching in normal mode")

            # load new articles
            self.dataset.load_new()

            last_id = self.dm.get_last_doc_id()
            latest = self.dataset.get_latest(last_id, content_type=index_by, filter_bl=enable_filtering)
            self.dm.add_docs(latest)

            print("Saving changes...", end="")
            self.dataset.save(res_path + "/dataset.pkl")
            self.dm.save(res_path + "/dtm.pkl")
            print("done")

            # init search engine
            print("Initializing search engine...", end="")
            search_bm25 = BM25(self.dm)
            print("done")

            do_analysis(self.dataset, search_bm25, latest, similarity_threshold)




def load_articles(res_path):
    """
    :param articles_path: path where the articles dump is stored
    :return: articles dataset
    """
    art_path = os.path.abspath(res_path) + "/dataset.pkl"
    print("Loading article dump...", end="")
    if not os.path.isfile(art_path):
        dataset = AData(res_path)
        dataset.load_new()
        dataset.save(art_path)
    dataset = AData.load(art_path)
    print("done. Articles in the dump: %d" % len(dataset.content))
    return dataset


def load_dtf(to_dtm_path, dataset, index_params):
    """
    :param dtm_path: path where the dtm dump is stored
    :param dataset: dataset object that is used to initialize dtm index
    :return: dtm index
    """
    dtm_path = os.path.abspath(to_dtm_path) + "/dtm.pkl"
    print("Loading document index...", end="")
    if not os.path.isfile(dtm_path):
        dm = WDM()
        dm.add_docs( dataset.get_latest(-1,
                                       content_type=index_params['index_by'],
                                       filter_bl=index_params['enable_filtering']) )
        dm.save(dtm_path)
    dm = WDM.load(dtm_path)
    print("done. Documents in the index: %d" % dm.get_n_docs() )
    return dm


def launch_server(dataset, search, HOST_NAME, PORT_NUMBER, res_path):
    server_class = HTTPServer
    httpd = server_class((HOST_NAME, PORT_NUMBER), nddHandler(dataset,search, res_path))
    print(time.asctime(), 'Server Starts - %s:%s' % (HOST_NAME, PORT_NUMBER))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print(time.asctime(), 'Server Stops - %s:%s' % (HOST_NAME, PORT_NUMBER))
    sys.exit()


def do_analysis(dataset, search, titles, similarity_threshold, mode="full"):
    # titles = dataset.get_titles()

    results = []

    for t_ord, title in enumerate(titles['docs']):
        compare_against = dataset.get_last_two_days(titles["ids"][t_ord])
        scores, ref = search.rank(title, compare_against)
        print(title)
        show = min(10, len(scores))
        similar = []
        for i in range(show):
            article_id = dataset.ids.index(ref[i])
            if titles["ids"][t_ord] != ref[i]:
                normalized_score = scores[i] / scores[0]
                if normalized_score < similarity_threshold:
                    break
                print(normalized_score, " ", dataset.titles[article_id])
                similar.append(dataset.ids[article_id])
        if len(similar)>0:
            results.append( {'article_id': titles["ids"][t_ord],
                            'similar_id': similar} )

    if len(results)>0:
        res_json = json.dumps(results, indent = 4)

        with open("./res/results.log","a") as r_log:
            r_log.write("%s\n\n" % time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime()))
            r_log.write(res_json)
            r_log.write("\n\n")
        post_json(res_json)


def main(params):

    # unpack parameters
    index_by = params['index_by']
    enable_filtering = params['enable_filtering']
    res_path = params['resources']
    # articles_dump = params['articles_dump']
    # dtm_dump = params['dtm_dump']
    similarity_threshold = params['similarity_threshold']
    server_mode = params['server_mode']

    # print("\n\nSpinning up...")
    # # load data
    # dataset = load_articles(res_path)
    # # print(len(dataset.content), "\n\n\n")
    #
    # # load index
    # dm = load_dtf(res_path, dataset, {'index_by': index_by, 'enable_filtering': enable_filtering})
    #
    # if server_mode:
    #     print("\nSwitching in server mode")
    #     # init search engine
    #     print("Initializing search engine...", end="")
    #     search_bm25 = BM25(dm)
    #     print("done")
    #     launch_server(dataset, search_bm25, HOST_NAME, PORT_NUMBER, res_path)
    #
    # else:
    #     print("\nSwitching in normal mode")
    #
    #     # load new articles
    #     dataset.load_new()
    #
    #     last_id = dm.get_last_doc_id()
    #     latest = dataset.get_latest(last_id, content_type=index_by, filter_bl=enable_filtering)
    #     dm.add_docs(latest)
    #
    #     print("Saving changes...", end="")
    #     dataset.save(res_path)
    #     dm.save(res_path)
    #     print("done")
    #
    #     # init search engine
    #     print("Initializing search engine...", end="")
    #     search_bm25 = BM25(dm)
    #     print("done")
    #
    #     do_analysis(dataset, search_bm25, latest, similarity_threshold)



if __name__ == "__main__":
    main()
