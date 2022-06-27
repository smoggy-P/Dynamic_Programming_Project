import numpy as np
from System import Plant
from Policy import Estimator
from LSTM import LSTM
import torch
import time
from arg_utils import get_args
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
np.random.seed(0)


class Experiment(object):
    def __init__(self, exp_cfg):
        self.exp_cfg = exp_cfg
        self.dt = exp_cfg.dt / 60 #hour
        self.T = exp_cfg.T
        self.n_sample = exp_cfg.n_sample
        self.std = exp_cfg.std
        self.c_dis = np.arange(start=0, stop=7.01, step=exp_cfg.c_resolution)
        self.p_dis = np.arange(start=0, stop=15.01, step=exp_cfg.p_resolation)
        self.l_dis = np.arange(start=0, stop=10.01, step=exp_cfg.l_resolution)
        self.u_dis = np.arange(start=-5, stop=4, step=exp_cfg.u_resolution)
        self.n = self.c_dis.shape[0] * \
            self.p_dis.shape[0] * self.l_dis.shape[0]

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def generate_transition_matrix(self):
        P_trans = {}
        plant = Plant(dt=self.dt)
        estimator = Estimator()

        for l_idx, l in enumerate(self.l_dis):
            l_hat = estimator.estimate(l)
            l_hats = np.random.normal(
                loc=l_hat, scale=self.std, size=self.n_sample)
            for p_idx, p in enumerate(self.p_dis):
                for c_idx, c in enumerate(self.c_dis):
                    for u_idx, u in enumerate(self.u_dis):
                        for l_hat in l_hats:
                            state_idx = self.get_idx(p_idx, c_idx, l_idx)
                            plant.reset(c, p, l)
                            plant.step(u, l_hat)
                            hit_state_idx_list = self.get_convex_combination(
                                plant.p, plant.battery.c, plant.l)
                            for hit_state_idx, hit_prob in hit_state_idx_list.items():
                                if P_trans.__contains__((state_idx, u_idx, hit_state_idx)):
                                    P_trans[state_idx, u_idx,
                                            hit_state_idx] += (1/self.n_sample) * hit_prob
                                else:
                                    P_trans[state_idx, u_idx, hit_state_idx] = (
                                        1/self.n_sample) * hit_prob
        return P_trans

    def get_idx(self, p_idx, c_idx, l_idx):
        '''
        encode p_idx, c_idx, l_idx to a single integer
        '''
        return int(p_idx * self.c_dis.shape[0] * self.l_dis.shape[0] +
                   c_idx * self.l_dis.shape[0] +
                   l_idx)

    def get_pcl(self, idx):
        '''
        decode p_idx, c_idx, l_idx from a single integer
        '''
        p_idx = idx // (self.c_dis.shape[0] * self.l_dis.shape[0])
        c_idx = (
            idx % (self.c_dis.shape[0] * self.l_dis.shape[0])) // self.l_dis.shape[0]
        l_idx = idx % self.l_dis.shape[0]
        return [self.p_dis[p_idx], self.c_dis[c_idx], self.l_dis[l_idx]]

    def close_idx(self, list, target):
        '''
        find closest index in list to target
        '''
        return np.argmin(np.abs(list - target))

    def idx_prob(self, dis, p):
        p_close_idx = self.close_idx(dis, p)
        if p >= dis[-1]:
            p_left_idx = dis.shape[0] - 1
            p_right_idx = dis.shape[0] - 2
            p_left_prob = 1
            p_right_prob = 0
        elif p <= 0:
            p_left_idx = 0
            p_right_idx = 1
            p_left_prob = 1
            p_right_prob = 0
        elif p >= dis[p_close_idx]:
            p_left_idx = p_close_idx
            p_right_idx = p_close_idx + 1
            p_right_prob = (p - dis[p_close_idx]) / \
                (dis[p_close_idx + 1] - dis[p_close_idx])
            p_left_prob = (dis[p_close_idx + 1] - p) / \
                (dis[p_close_idx + 1] - dis[p_close_idx])
        elif p < dis[p_close_idx]:
            p_left_idx = p_close_idx - 1
            p_right_idx = p_close_idx
            p_right_prob = (p - dis[p_close_idx - 1]) / \
                (dis[p_close_idx] - dis[p_close_idx - 1])
            p_left_prob = (dis[p_close_idx] - p) / \
                (dis[p_close_idx] - dis[p_close_idx - 1])
        return [p_left_idx, p_right_idx, p_left_prob, p_right_prob]

    def get_convex_combination(self, p, c, l):
        """Get convex combination given target state

        Returns:
            dict: dictionary of hit state and probability
        """

        hit_state_idx_list = {}
        p_idx = np.zeros(2)
        p_prob = np.zeros(2)
        c_idx = np.zeros(2)
        c_prob = np.zeros(2)
        l_idx = np.zeros(2)
        l_prob = np.zeros(2)

        p_idx[0], p_idx[1], p_prob[0], p_prob[1] = self.idx_prob(self.p_dis, p)
        c_idx[0], c_idx[1], c_prob[0], c_prob[1] = self.idx_prob(self.c_dis, c)
        l_idx[0], l_idx[1], l_prob[0], l_prob[1] = self.idx_prob(self.l_dis, l)

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    hit_idx = self.get_idx(p_idx[i], c_idx[j], l_idx[k])
                    hit_prob = p_prob[i] * c_prob[j] * l_prob[k]
                    hit_state_idx_list[hit_idx] = hit_prob

        return hit_state_idx_list

    def dynamic_programming(self, P_trans):
        P_trans_tensor = torch.sparse_coo_tensor(indices=list(zip(*P_trans.keys())), values=list(
            P_trans.values()), size=(self.n, self.u_dis.shape[0], self.n), device=self.device, dtype=torch.float)
        J = torch.tensor([self.get_pcl(i)[0]
                         for i in range(self.n)], device=self.device).float()
        Actions = torch.zeros(size=(self.T, self.n), device=self.device)

        for t in range(self.T-1, 0, -1):
            print("DP step: {}".format(t))
            J_update = torch.zeros_like(J, device=self.device)
            for i in range(self.n):
                J_update[i], Actions[t][i] = torch.min(
                    P_trans_tensor[i] @ J, dim=0)
            J = J_update
        return Actions

    def run(self):
        print("Generating transition matrix ...")
        start_time = time.time()
        P_trans = self.generate_transition_matrix()
        print("Transition matrix finished, takes time: {:.3}s".format(time.time()-start_time))
        
        print("Doing DP ...")
        start_time = time.time()
        Actions = self.dynamic_programming(P_trans)
        print("DP finished, takes time: {:.3}s".format(time.time()-start_time))
        
        if self.exp_cfg.save_action_table == True:
            torch.save(Actions.to(device='cpu'), 'saved_models/Action_table_new.pt')



if __name__ == "__main__":
    exp_cfg = get_args()
    experiment = Experiment(exp_cfg)
    experiment.run()
