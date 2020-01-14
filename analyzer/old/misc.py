import os

def merge_articles(art):
 c = ""
 for v in art.values():
  c += v['body'] + "\n"
 return c

def create_folders(path):
    resource_folder = os.path.join(path,'resources')
    print(resource_folder)
    if not os.path.isdir(resource_folder):
        os.mkdir(resource_folder)
