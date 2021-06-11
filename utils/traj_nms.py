import numpy as np

def traj_nms(traj, score, threshold=2.0):
    '''
        traj:[K, 30, 2] list
        score:[K]
    '''
    traj = np.array(traj)
    score = np.array(score)
    sorted_index = np.argsort(score)[::-1] #from max to min

    candidate = []
    candidate_score = []
  
    while len(sorted_index):
        index = sorted_index[0]        
        curr_traj = traj[index]
        candidate.append(curr_traj)
        candidate_score.append(score[index])
        if len(sorted_index)<=1 or len(candidate)>=6:
            break
        distance = np.linalg.norm(curr_traj[-1,:]-traj[sorted_index[1:]][:,-1,:], 2, axis=-1)
        new_index = np.where(distance>threshold)[0]#the first one
        sorted_index = sorted_index[new_index+1]
    
    return candidate, candidate_score

# def traj_nms_batch(traj, score, threshold=2.0):
#     '''
#         traj:[N, K, 30, 2] 
#         score:[N, K]
#     '''
#     sorted_index = np.argsort(score)[:,::-1] #from max to min

#     candidate = np.zeros_like(traj)
#     candidate_score = np.zeros_like(score)
  
#     while len(sorted_index):
#         index = sorted_index[0]        
#         curr_traj = traj[index]
#         candidate.append(curr_traj)
#         candidate_score.append(score[index])
        
#         if len(sorted_index)<=1 or len(candidate)>=6:
#             break
#         distance = np.linalg.norm(curr_traj[-1,:]-traj[sorted_index[1:]][:,-1,:], 2, axis=-1)
#         new_index = np.where(distance>threshold)[0]#the first one
#         sorted_index = sorted_index[new_index+1]
    
#     return candidate, candidate_score

def traj_snms(traj, score, threshold=2.0):
    '''
        traj:[K, 30, 2] list
        score:[K]
    '''
    traj = np.array(traj)
    score = np.array(score)
    origial_score = score.copy()
    sorted_index = np.argsort(score)[::-1] #from max to min
    score_sorted_index = np.argsort(score)[::-1]
    candidate = []
    candidate_score = []

    real_traj = (traj[score_sorted_index][:6]).copy()

    while len(sorted_index):
        index = sorted_index[0] 

        curr_traj = traj[index]  
        candidate.append(traj[index])
        candidate_score.append(score[score_sorted_index[0]])
        if len(sorted_index)<=1 or len(candidate)>=6:
            break

        distance = np.linalg.norm(curr_traj[-1,:]-traj[sorted_index[1:]][:,-1,:], 2, axis=-1)
        under_threshold_mask = distance < 4.0
        iou = get_iou(distance*under_threshold_mask)
        iou = iou*under_threshold_mask
        score = score[score_sorted_index[1:]]\
                    * np.exp(-iou**2*(2))
        score_sorted_index = np.argsort(score)[::-1]
        sorted_index = sorted_index[1:][score_sorted_index]
  

    return candidate, candidate_score

def get_iou( distance):

    intersction = 16*np.pi*np.arccos(distance/4) - distance*np.sqrt(4-(distance**2/4))
    union = 2**2*np.pi*2 - intersction
    iou = intersction/union
    return iou


def traj_nms_with_region_index(traj, score, number_region=6, threshold=2.0):
    '''
        traj:[K, 30, 2] list
        score:[K]
    '''
    traj = np.array(traj)
    score = np.array(score)
    sorted_index = np.argsort(score)[::-1] #from max to min

    K = traj.shape[0]
    region_proposal_num = K//number_region
    candidate = []
    candidate_region_index = []
  
    while len(sorted_index):
        index = sorted_index[0]  
        region_index = index//K    
        curr_traj = traj[index]
        candidate.append(curr_traj)
        candidate_region_index.append(region_index)
        if len(sorted_index)<=1 or len(candidate)>=6:
            break
        distance = np.linalg.norm(curr_traj[-1,:]-traj[sorted_index[1:]][:,-1,:], 2, axis=-1)
        new_index = np.where(distance>threshold)[0] # the first one
        sorted_index = sorted_index[new_index+1]
    

 
    return candidate


# ADJACENT_MATRIX = np.array([ [2,4],
#                     [3,5], 
#                     [2,0],
#                     [1,],
#                     [0,5],
#                     [1,4]])


def traj_region_nms(traj, score, threshold=2.0):
    '''
        traj:[K, 30, 2] list
        score:[K]
    '''
    traj = np.array(traj)
    score = np.array(score)
    sorted_index = np.argsort(score)[::-1] #from max to min

    candidate = []
    candidate_score = []
    

    best_region_index = np.argmax(traj.reshape(6,6,30,2).sum(1))
    


    while len(sorted_index):
        index = sorted_index[0]        
        curr_traj = traj[index]
        candidate.append(curr_traj)
        candidate_score.append(score[index])
        if len(sorted_index)<=1 or len(candidate)>=6:
            break
        distance = np.linalg.norm(curr_traj[-1,:]-traj[sorted_index[1:]][:,-1,:], 2, axis=-1)
        new_index = np.where(distance>threshold)[0]#the first one
        sorted_index = sorted_index[new_index+1]
    
    return candidate, candidate_score


def plot_f():
    x = np.arange(0,10,0.1)
    y = np.exp((x))
    import matplotlib.pyplot as plt

    plt.plot(x,y, '--')
    plt.show()

import pickle
import matplotlib.path as path
from .evaluation_utils import evaluate
from tqdm import tqdm

class NMS_withregion(object):

    def __init__(self, map_name, K, threshold=1.5, NMS_max_num=6):
        
        self.threshold = threshold
        self.NMS_max_num = NMS_max_num

        map_name = './partition_map/{}.pkl'.format(map_name)

        with open(map_name, 'rb') as file:
            self.endpoints_map = pickle.load(file)
       
        self.regions = []
        bias = 0.001
        for _, region in enumerate(self.endpoints_map):
            self.regions.append(path.Path(region+bias))

        self.num_region = len(self.regions)
        self.N = K//self.num_region
        self.K = K

    def NMS(self, traj, score):

        traj = np.array(traj)
        score = np.array(score)
        sorted_index = np.argsort(score)[::-1] #from max to min

        candidate = []
        candidate_score = []
        candidate_index = []
        while len(sorted_index):
            index = sorted_index[0]        
            curr_traj = traj[index]
            candidate_index.append(index)
            candidate.append(curr_traj)
            candidate_score.append(score[index])
            if len(sorted_index)<=1 or len(candidate)>=self.NMS_max_num:
                break
            distance = np.linalg.norm(curr_traj[-1,:]-traj[sorted_index[1:]][:,-1,:], 2, axis=-1)
            new_index = np.where(distance>self.threshold)[0]#the first one
            sorted_index = sorted_index[new_index+1]

        return candidate, candidate_score, candidate_index

    def NMS_condition_on_selected_trajs(self, traj, score, selected_trajs):

        """
            params:
                trajs: [N,30,2]
                probs: [N,]
                selected_trajs: list[M * [30, 2]]

           
        """
        #filter the trajs inside selected_trajs scope
        for i in range(len(selected_trajs)):
            dist = np.linalg.norm(selected_trajs[i][-1,:]-traj[:,-1,:], 2, axis=-1)
            remain_index = np.where(dist>self.threshold)
            traj, score = traj[remain_index], score[remain_index]
     
        sorted_index = np.argsort(score)[::-1] #from max to min

        candidate = []
        candidate_score = []
        while len(sorted_index):
            index = sorted_index[0]        
            curr_traj = traj[index]
            candidate.append(curr_traj)
            candidate_score.append(score[index])
            if len(sorted_index)<=1 or len(candidate)>=(self.NMS_max_num-len(selected_trajs)):
                break
            distance = np.linalg.norm(curr_traj[-1,:]-traj[sorted_index[1:]][:,-1,:], 2, axis=-1)
            new_index = np.where(distance>self.threshold)[0]#the first one
            sorted_index = sorted_index[new_index+1]

        candidate.extend(selected_trajs)
        return candidate, candidate_score

    def region_select_vote(self, trajs, probs, TopK=6):
        """
            params:
                trajs: [N,30,2]
                probs: [N,]

            choosing the region based on Top 6 region region/
            voting by all
        """
        trajs_endpoint = trajs[:,-1,:]
        # trajs_endpoint_dist = np.linalg.norm(trajs_endpoint, ord=2, axis=-1) #[N]

        region_ballot = []
        for region in self.regions:
            inside_region_num = region.contains_points(trajs_endpoint).sum()
            region_ballot.append(inside_region_num)
        

        region_index = region_ballot.index(max(region_ballot))
        return region_index

    def region_select_topK(self, trajs, probs, TopK=6):
        """
            params:
                trajs: [N,30,2]
                probs: [N,]

            choosing the region based on Top 6 region region/
            voting by all
        """
        topK_index = np.argsort(probs)[::-1][:TopK]
        trajs_endpoint = trajs[topK_index,-1,:]
        # trajs_endpoint_dist = np.linalg.norm(trajs_endpoint, ord=2, axis=-1) #[N]
        
        region_ballot = []
        mean_endpoint =  trajs_endpoint.mean(axis=0)
        region_index = -1
        for i, region in enumerate(self.regions):
            inside_region = region.contains_point(mean_endpoint)
            if inside_region:
                region_index = i
            # region_ballot.append(inside_region_num)

        for region in self.regions:
            inside_region_num = region.contains_points(trajs_endpoint).sum()
            region_ballot.append(inside_region_num)
        
        # self.probs_select_region.append(probs[topK_index])
        region_index_old = region_ballot.index(max(region_ballot))
        # if region_index_old!=region_index:
            # import pdb; pdb.set_trace()
        return region_index

    def region_select_mean(self, trajs, probs, TopK=6):
        """
            params:
                trajs: [N,30,2]
                probs: [N,]

            choosing the region based on Top 6 region region/
            voting by all
        """
        trajs_endpoint = trajs[:,-1,:]
        # trajs_endpoint_dist = np.linalg.norm(trajs_endpoint, ord=2, axis=-1) #[N]
        new_trajs = []
        new_probs = []
        for i, region in enumerate(self.regions):
            # mean_endpoint = trajs_endpoint[i*self.N:(i+1)*self.N].mean(axis=0)
            # best_case_index = np.argmax(probs[i*self.N:(i+1)*self.N])
            mean_endpoint = trajs_endpoint[i*self.N:(i+1)*self.N]
            index_good_case = np.nonzero(region.contains_points(mean_endpoint))[0]+i*self.N
            new_trajs.append(trajs_endpoint[index_good_case])
            new_probs.append(probs[index_good_case])    
        new_trajs = np.concatenate(new_trajs, axis=0)
        new_probs = np.concatenate(new_probs, axis=0)
        best_endpoints = new_trajs[np.argmax(new_probs)]
        region_index =-1 
        for i, region in enumerate(self.regions):
            inside_region = region.contains_point(best_endpoints)
            if inside_region:
                region_index = i

        return region_index
    

    def process(self, trajs, probs, gt, case_index):
        try:
            gt_index = [region.contains_point(gt[-1,:]) for region in self.regions].index(True)
        except:
            gt_index = -1
        # index2 = self.region_select_topK(trajs, probs, 6)
        # index = self.region_select_mean(trajs, probs, 6)
        
        # index2 = self.region_select_vote(trajs, probs)
        # if index!=index2: 
        #     if index==gt_index or index2==gt_index:
        #         self.right_num+=1
        #         index = gt_index
        # index = gt_index if gt_index!=
        # select_trajs_index = slice(index*self.N, (index+1)*self.N)
        # if index== gt_index: 
        #     self.right_num+=1
        #     fde = np.linalg.norm(gt[-1,:]-trajs[select_trajs_index], ord=2, axis=-1) < 2
        #     if fde.sum() > 0:
        #         self.right_predict +=1 
        # else:
        #     self.region_confuse[gt_index, index] += 1
        #     # self.Bad_case_index.append(case_index)
        #     # self.probs_select_region.pop()
        # # if gt_index!=-1:
        # #     index=gt_index
        # # import pdb; pdb.set_trace()
        # select_trajs, _, _ = self.NMS(trajs[select_trajs_index], probs[select_trajs_index])
        # final_trajs, _ = self.NMS_condition_on_selected_trajs(trajs, probs, select_trajs)
        if gt_index == -1:
            return None
        index = gt_index 
        # if case_index == 294:
        #     import pdb; pdb.set_trace()
        select_trajs_index = slice(index*self.N, (index+1)*self.N)
        fde_self = np.linalg.norm(gt[-1,:]-trajs[select_trajs_index][:,-1,:], ord=2, axis=-1) < 2
        # select_trajs, _, _ = self.NMS(trajs, probs)
        fde_all = np.linalg.norm(gt[-1,:]-trajs[:,-1,:], ord=2, axis=-1) < 2
        if fde_self.sum()==0 and fde_all.sum() == 0:
        # if  gt[-1,1] < 0:
            # if case_index==55:
            #     import pdb; pdb.set_trace()
            self.Bad_case_index.append(case_index)
        return None
        # return final_trajs
    
    def evaluate(self, trajs, probs , gt_trajectories):
        forecasted_trajectories, forecast_prob = {}, {}
        self.right_num =0
        self.right_predict = 0
        self.Bad_case_index = []
        self.region_confuse = np.zeros((self.num_region, self.num_region))
        self.gt = gt_trajectories
        for i, _ in tqdm(enumerate(trajs)):
            forecasted_trajectories[i] =\
                self.process(trajs[i][:], probs[i][:], gt_trajectories[i], i)    

        with open('./selfwrongotherright.pkl', 'wb') as file:
            pickle.dump(self.Bad_case_index,file)
        print(len(self.Bad_case_index))
        # print('AC:',(self.right_predict/self.right_num))
        # print('right_num:',self.right_predict,self.right_num)
        # print(self.region_confuse.astype(np.int))
        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # sns.heatmap(self.region_confuse)
        # plt.show()
        # # print(sum(self.probs_select_region)/len(self.probs_select_region))
        # result = evaluate( forecasted_trajectories, 
        #                     gt_trajectories, 
        #                     # forecasted_probabilities = forecast_prob
        #                     )
        result = '1'
        return result


if __name__ == "__main__":
    # plot_f()

    test_point = np.array([ 1, 2])
    print(get_iou(test_point))