import sys, os, pickle
from tqdm import tqdm

cache_root = sys.argv[1]

car = 0
ped = 0
cyc = 0

if os.path.exists(cache_root):
    g = os.walk(cache_root)
    for path, dir_list, file_list in g:
        if 'train' not in path:
            continue
        print(path)
        progress = tqdm(file_list)
        cnt = len(file_list)
        for file_name in progress:
            with open(os.path.join(path, file_name), 'rb') as f:
                data = pickle.load(f)
                obj_type = data['obj_type'][data['tracks_to_predict']]
                if 1 in obj_type:
                    car += 1
                if 2 in obj_type:
                    ped += 1
                if 3 in obj_type:
                    cyc += 1
                    for i range(4):
                        with open(os.path.join(path, f'{cnt}.pkl'), 'wb') as ff:
                            pickle.dump(data, dump)
                            cnt += 1
                            print(f'new file {cnt}.pkl!')
            progress.set_description(desc=f'car-{car}-ped-{ped}-cyc-{cyc}')
else:
    print(f'{cache_root} not exists!')
print(car, ped, cyc)
