import pickle

articles = pickle.load(open("articles.dat","rb"))

ids = dict(zip(range(len(articles)),articles.keys()))

NEIGHBORS = 10

with open("doc_dist_kl.txt","r") as kl_dist:
    for line in kl_dist.readlines():
        dists = [float(i) for i in line.strip().split(" ")]
        doc = int(dists[0])
        dists = dists[1:]
        dists[dists.index(min(dists))] = 100.

        print("=======DOCUMENT============")
        print("\t\t",articles[ids[doc]]['title'])
        print(articles[ids[doc]]['body'])
        print("\n")

        for i in range(NEIGHBORS):
                min_id = dists.index(min(dists))

                print("=======CLOSEST %d============"%i)
                print("\t\t",articles[ids[min_id]]['title'])
                print(articles[ids[min_id]]['body'])
                print("\n\n\n")
                dists[min_id] = 100.
