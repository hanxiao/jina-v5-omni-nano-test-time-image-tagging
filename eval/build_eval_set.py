"""Build a real multi-label eval set from COCO val2017.
Sample N images, record their COCO category names (multi-label GT), download.
"""
import json, os, urllib.request, concurrent.futures as cf, random
HERE=os.path.dirname(__file__)
ann=json.load(open(os.path.join(HERE,"annotations/instances_val2017.json")))
cats={c["id"]:c["name"] for c in ann["categories"]}
# image_id -> set of category names
from collections import defaultdict
img2cats=defaultdict(set)
for a in ann["annotations"]:
    img2cats[a["image_id"]].add(cats[a["category_id"]])
imgs={im["id"]:im for im in ann["images"]}
random.seed(42)
ids=random.sample([i for i in img2cats if len(img2cats[i])>=1], 150)
os.makedirs(os.path.join(HERE,"eval_imgs"), exist_ok=True)
gt={}
def dl(iid):
    im=imgs[iid]; fn=f"{iid}.jpg"; p=os.path.join(HERE,"eval_imgs",fn)
    try:
        if not os.path.exists(p):
            urllib.request.urlretrieve(im["coco_url"], p)
        return iid, sorted(img2cats[iid])
    except Exception: return iid, None
with cf.ThreadPoolExecutor(12) as ex:
    for iid,g in ex.map(dl, ids):
        if g: gt[str(iid)]=g
json.dump(gt, open(os.path.join(HERE,"eval_gt.json"),"w"))
print("eval set:", len(gt), "images")
allc=set(); [allc.update(v) for v in gt.values()]
print("categories present:", len(allc))
import numpy as np
print("avg labels/img:", np.mean([len(v) for v in gt.values()]).round(2))
