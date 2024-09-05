import json
import pickle
from tqdm import tqdm
import time

from parse_hh_data.parse_hh_data import download, parse

resume_ids = pickle.load(open("parsed_links.pkl", "rb"))
resume_ids = [i for i in resume_ids if len(i) == 38]
print("Finded resumes count: ", len(resume_ids))

results = []

for i, ids in enumerate(tqdm(resume_ids)):
    if ids == "programmist":
        continue
    try:
        a = parse.resume(download.resume(ids))
        results.append(a)
        time.sleep(1)
    except:
        continue

    if i % 100 == 1 or i == len(resume_ids) - 1:
        with open("programmist_hh_resumes.pkl", "wb") as f:
            pickle.dump(results, f)
