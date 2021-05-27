import sys, os, pickle
from tqdm import tqdm

cache_root = sys.argv[1]

car = 0
ped = 0
cyc = 0

if os.path.exists(cache_root):
    g = os.walk(cache_root)
    for path, dir_list, file_list in g:
        for file in tqdm(file_list):
            with open(os.path.join(path, file), 'rb') as f:
                data = pickle.load(f)
                obj_type = data['obj_type'][data['tracks_to_predict']]
                if 1 in obj_type:
                    car += 1
                if 2 in obj_type:
                    ped += 1
                if 3 in obj_type:
                    cyc += 1
else:
    print(f'{cache_root} not exists!')
print(car, ped, cyc)