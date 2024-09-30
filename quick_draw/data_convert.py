import pandas as pd
import numpy as np
import pickle
import glob

bum_points = 2048

files = glob.glob("data\exp\simplified/ndjson/*.ndjson")

all_data = []
for file in files:
    data = pd.read_json(file, lines=True)
    data = data["drawing"]
    for i in data:
        tmp = []
        for j in i:
            tmp.append(np.array(j, dtype=float).T.tolist())
        all_data.append(tmp)

with open('data\exp\simplified/all/list.pickle', mode='wb') as f:
    pickle.dump(all_data, f)