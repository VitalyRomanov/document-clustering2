# import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from copy import deepcopy
import json
import pickle as p
import os

HOST_NAME = 'localhost'
PORT_NUMBER = 9000
results_path = "../res/CompRes.dat"

id_next = 0
init_complete = False


def init_id_next(cr):
    global id_next
    id_next = cr.last
    print("Moving pointer to article #%d"%id_next)


class CompResult:

    results = None
    last = 0

    def __init__(self):
        results = dict()

    def track_res(self,res):
        # print("tracking",res['id'],res['similar'])
        self.results[res['id']] = res['similar']

    def load(path):
        return p.load(open(path,"rb"))

    def save(self,path):
        # print("saving")
        p.dump(self, open(path,"wb"))

    def has_res(self,c_id):
        # print("checking %s" % c_id)
        return c_id in self.results.keys()

    def get_results(self,c_id):
        return self.results[c_id]


def create_compres():
    global results_path
    if not os.path.isfile(results_path):
        cr = CompResult()
        cr.save(results_path)


def nddHandler(doc_dataset, doc_search, resource_path):
    # init to the last labeled document
    global results_path
    results_path = os.path.abspath(resource_path) + "/CompRes.dat"
    create_compres()
    results = CompResult.load(results_path)
    init_id_next(results)

    # create handler class
    class NddHandler(BaseHTTPRequestHandler):
        dataset = doc_dataset
        search = doc_search
        res_path = resource_path

        def configure(self,titles):
            self.titles = deepcopy(titles)

        def do_HEAD(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

        def do_GET(self):

            request = self.path[1:]

            try:
                a_id = int(request)

            except ValueError:
                global id_next
                if request == 'next':
                    id_next = min(id_next+1,len(self.dataset.ids))
                    a_id = id_next
                    print('Selecting next',a_id, self.dataset.ids[a_id])
                elif request == 'prev':
                    id_next = max(id_next-1,0)
                    a_id = id_next
                    print('Selecting prev',a_id, self.dataset.ids[a_id])

            title = self.dataset.titles[a_id]
            article_content = self.dataset.content[a_id]
            article_id = self.dataset.ids[a_id]


            results = CompResult.load(results_path)

            if results.has_res(article_id):
                p_res = results.get_results(article_id)
            else:
                p_res = False

            similar, human = self.do_search(title,p_res)
            content = {'title':title,
                        'content':article_content,
                        'id':article_id,
                        'similar':similar,
                        'human': 1 if human else 0}

            self.respond({'status': 200},content)


        def do_POST(self):
            request = self.path[1:]
            self.data_string = self.rfile.read(int(self.headers['Content-Length']))
            data = json.loads(self.data_string)

            results = CompResult.load(results_path)
            results.track_res(data)
            results.last = id_next
            results.save(results_path)
            # print(data)
            try:
                content = "ok"
                self.respond({'status': 200},content)
            except:
                pass


        def handle_http(self, status_code, a_content):
            self.send_response(status_code)
            self.send_header('Content-type', 'text/html')
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()


            if isinstance(a_content,dict):
                to_send = json.dumps(a_content,indent=4)
                content = to_send

            return bytes(content, 'UTF-8')

        def respond(self, opts, content):
            response = self.handle_http(opts['status'], content)
            self.wfile.write(response)

        def do_search(self,query, p_results):
            scores, ref = self.search.rank(query)
            show = min(11, len(scores))
            similar = []

            if p_results == False:
                override = False
            else:
                print("Human labeling available", p_results)
                override = True
            for i in range(1,show):
                article_id = self.dataset.ids.index(ref[i])
                artcl = {}
                normalized_score = scores[i] / scores[0]
                artcl['title'] = "%.2f" % (normalized_score) + " " + self.dataset.titles[article_id]
                artcl['content'] = self.dataset.content[article_id]
                artcl['match'] = 1 if normalized_score > .4 else 0
                if override:
                    artcl['match'] = 1 if ref[i] in p_results else 0
                artcl['id'] = self.dataset.ids[article_id]
                similar.append(artcl)
            return similar, override

    return NddHandler
