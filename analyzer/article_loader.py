import json
import os
import joblib as p
from datetime import datetime
import urllib.request
import numpy as np


def date2int(date):
    return int(datetime.strptime(date, '%Y-%m-%d  %H:%M:%S').timestamp())


def get_date(ts):
    return datetime.fromtimestamp(
        int(repr(ts))
    ).strftime('%Y-%m-%d %H:%M:%S')


def load_latest():
    dump_file = "articles_dump.dat"
    l_time = 1509031277
    if os.path.isfile(dump_file):
        articles = p.load(open(dump_file, "rb"))
    else:
        articles = []
    return articles


# def retreive_articles(l_time):
#     data = json.load(open('1509031277.json'))
#     # retreive articles' dates
#     dates = list(map(date2int, map(lambda x: x['public_date'], data)))
#     # sort articles by date
#     s_ind = sorted(range(len(dates)), key=lambda k: dates[k])
#     s_data = [data[ind] for ind in s_ind]
#     return s_data


def retreive_articles_url(time):
    """
    :param time: the last available record, encodes time as integer
    :return: list of article records sorted by date
    """
    url_addr = "https://www.business-gazeta.ru/index/monitoring/timestamp/%d" % time
    data = None
    with urllib.request.urlopen(url_addr) as url:
        data = json.loads(url.read().decode())

    dates = list(map(date2int, map(lambda x: x['public_date'], data)))

    # sort articles by date
    s_ind = sorted(range(len(dates)), key=lambda k: dates[k])
    s_data = [data[ind] for ind in s_ind]
    return s_data


def post_json(data_json):
    url_addr = "https://www.business-gazeta.ru/index/similar"
    enc_json = data_json.encode('utf-8')
    req = urllib.request.Request(url_addr, data=enc_json,
                                 headers={'content-type': 'application/json'})
    response = urllib.request.urlopen(req)
    print(response.read())


# def get_last_time(articles):
#     return articles[-1] if len(articles) != 0 else 0
#     latest = 0
#     for article in articles:
#         candidate = date2int(article['public_date'])
#         if candidate > latest:
#             latest = candidate
#     return latest


def get_sections(s_data):
    # split data into sections
    ids = list(map(lambda x: x['id'], s_data))
    titles = list(map(lambda x: x['title'], s_data))
    content = list(map(lambda x: x['content'], s_data))
    dates = list(map(date2int, map(lambda x: x['public_date'], s_data)))
    links = list(map(lambda x: x['link'], s_data))
    return ids, titles, content, dates, links


class AData:
    ids = None
    titles = None
    content = None
    dates = None
    links = None
    _TWO_DAYS = 60 * 60 * 24 * 2  # sec*min*hr*2d

    def __init__(self, res_path):
        self.ids = []
        self.titles = []
        self.content = []
        self.dates = []
        self.links = []
        self.res_path = res_path

        articles_data = get_sections(load_latest())
        self.join_sections(articles_data)
        # self._latest = self.get_last_time()
        # self.new = len(self.ids)

    def load_new(self):
        """
        Load new articles from the server
        """
        latest = self.get_last_time()
        last_id = self.ids[-1] if len(self.ids)>0 else ""

        # Log the event
        log_string = "Retreiving after %s with id %s: " % ( get_date(latest), last_id)
        print(log_string, end="")
        with open(self.res_path + "/retrieving.log","a") as r_log:
            r_log.write(log_string)

        new_articles = retreive_articles_url(latest)
        self.join_sections( get_sections(new_articles) )
        new = len(new_articles)

        # Log the event result
        if new == 0:
            log_string = "Nothing new"
        else:
            log_string = "%d added" % new
        print(log_string)
        with open(self.res_path + "/retrieving.log","a") as r_log:
            r_log.write(log_string+"\n")

    def join_sections(self, articles_data):
        ids, titles, content, dates, links = articles_data
        self.ids += ids
        self.titles += titles
        self.content += content
        self.dates += dates
        self.links += links

    def get_article(self, a_id):
        return self.content[a_id]

    def get_last_time(self):
        return self.dates[-1] if (len(self.dates) > 0) else 1518269083#1509031277

    def two_days_range(self, id1, id2):
        return True if abs(self.dates[id1] - self.dates[id2]) < self._TWO_DAYS else False

    def get_last_two_days(self, a_id):
        begin_with = self.ids.index(a_id)
        ids = []
        for i in range(begin_with, -1, -1):
            if self.two_days_range(begin_with, i):
                ids.append(self.ids[i])
            else:
                break
        return np.array(ids)

    def make_json(self, doc_id, similar_id):
        return json.dumps({"article_id": self.ids[doc_id],
                           "similar_id": [self.ids[s_id] for s_id in similar_id]},
                          indent=4)

    def get_latest(self, last_id, content_type='titles', filter_bl = True):
        """
        Input:  last_id - the id in self.ids.
                content_type - optional. Specifies whether to return titles or articles'
                body
                filter_bl - specifies whether to apply blacklist filtering or not
        Returns: all documents and ids that appear after the doc with last_id
        """
        try:
            last_pos = self.ids.index(last_id)
        except:
            if last_id != -1:
                raise Exception("No document with such id")
            last_pos = last_id

        if content_type == 'titles':
            content_source = self.titles
        elif content_type == 'content':
            content_source = self.content
        else:
            raise NotImplemented

        latest_ids = []
        latest_content = []
        for i in range(last_pos + 1, len(self.ids)):
            if filter_bl and self.is_blacklisted(i):
                continue
            latest_ids.append(self.ids[i])
            latest_content.append(content_source[i])

        return {'ids': latest_ids, 'docs': latest_content}

    def get_last_titles(self, last_n=-1):
        """
        :param last_n: the number of latest titles to return
        :return: dictionary that contains ids and the content of titles
        """
        titles_total = len(self.titles)
        if last_n == -1:
            titles_range = range(titles_total)
        else:
            titles_range = range(max(titles_total - last_n, 0), titles_total)

        titles_ids = []
        titles_content = []
        for i in titles_range:
            if not self.is_blacklisted(i):
                titles_ids.append(self.ids[i])
                titles_content.append(self.titles[i])

        return {'ids': titles_ids, 'titles': titles_content}

    def is_blacklisted(self, ind: int) -> bool:
        black_list = ['realnoevremya.ru', 'tatcenter.ru']
        url = self.links[ind].split("/")[2]
        return url in black_list

    def load(path):
        return p.load(open(path, "rb"))

    def save(self,path):
        p.dump(self, open(path, "wb"))
