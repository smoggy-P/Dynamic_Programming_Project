import numpy as np
from LSTM import LSTM
from System import Plant
from Policy import Estimator
import warnings
from tqdm import tqdm
import torch

def close_idx(list, target):
    '''
    find closest index in list to target
    '''
    return np.argmin(np.abs(list - target))

def get_idx(p_idx, c_idx, l_idx):
    '''
    encode p_idx, c_idx, l_idx to a single integer
    '''
    return int(p_idx * c_dis.shape[0] * l_dis.shape[0] + 
            c_idx * l_dis.shape[0]                  +
            l_idx)
    
def get_pcl(idx):
    '''
    decode p_idx, c_idx, l_idx from a single integer
    '''
    p_idx = idx // (c_dis.shape[0] * l_dis.shape[0])
    c_idx = (idx % (c_dis.shape[0] * l_dis.shape[0])) // l_dis.shape[0]
    l_idx = idx % l_dis.shape[0]
    return [p_dis[p_idx], c_dis[c_idx], l_dis[l_idx]]

def get_convex_combination(p, c, l):
    """Get convex combination given target state

    Returns:
        dict: dictionary of hit state and probability
    """
    
    hit_state_idx_list = {}
    
    def idx_prob(dis, p):
        p_close_idx = close_idx(dis, p)
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
            p_right_prob = (p - dis[p_close_idx]) / (dis[p_close_idx + 1] - dis[p_close_idx])
            p_left_prob = (dis[p_close_idx + 1] - p) / (dis[p_close_idx + 1] - dis[p_close_idx])
        elif p < dis[p_close_idx]:
            p_left_idx = p_close_idx - 1
            p_right_idx = p_close_idx
            p_right_prob = (p - dis[p_close_idx - 1]) / (dis[p_close_idx] - dis[p_close_idx - 1])
            p_left_prob = (dis[p_close_idx] - p) / (dis[p_close_idx] - dis[p_close_idx - 1])
        return [p_left_idx, p_right_idx, p_left_prob, p_right_prob]
    
    p_idx = np.zeros(2)
    p_prob = np.zeros(2)
    c_idx = np.zeros(2)
    c_prob = np.zeros(2)
    l_idx = np.zeros(2)
    l_prob = np.zeros(2)
    
    p_idx[0], p_idx[1], p_prob[0], p_prob[1] = idx_prob(p_dis, p)
    c_idx[0], c_idx[1], c_prob[0], c_prob[1] = idx_prob(c_dis, c)
    l_idx[0], l_idx[1], l_prob[0], l_prob[1] = idx_prob(l_dis, l)
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                hit_idx = get_idx(p_idx[i], c_idx[j], l_idx[k])
                hit_prob = p_prob[i] * c_prob[j] * l_prob[k]
                hit_state_idx_list[hit_idx] = hit_prob
    
    return hit_state_idx_list

warnings.filterwarnings('ignore')
np.random.seed(0)

c_resolution = 0.5 #unit: kW 
p_resolation = 3 #unit: kW
l_resolution = 5 #unit: kW
u_resolution = 1 # TODO
dt = 1/60 # hour
T  = 2
plant = Plant(dt=dt)
c_dis = np.arange(start=0, stop=7.01, step=c_resolution) # TODO kW`h should has larger resolution
p_dis = np.arange(start=0, stop=15.01, step=p_resolation) 
l_dis = np.arange(start=0, stop=10.01, step=l_resolution)
u_dis = np.arange(start=-5, stop=4, step=u_resolution)

n = c_dis.shape[0] * p_dis.shape[0] * l_dis.shape[0]
print("Size of state space: {}".format(n))
print("Size of action space: {}".format(u_dis.shape[0]))

device = torch.device('cpu')

# P_trans_tensor = torch.sparse_coo_tensor(indices=list(zip(*P_trans.keys())), values=list(P_trans.values()), size=(n, u_dis.shape[0], n), device=device, dtype=torch.float)
J = torch.exp(torch.tensor([get_pcl(i)[0] for i in range(n)], device=device).float())
Actions = torch.zeros(size=(T, n), device=device)
for t in tqdm(range(T-1, -1, -1)):
    J_update = torch.zeros_like(J)
    for i in (range(n)): 
        p, c, l = 7, 0, 1
        J_expect = torch.zeros(u_dis.shape[0])
        for u_idx, u in enumerate(u_dis):
            plant.reset(c, p, l)
            plant.step(u, l)
            hit_state_list = get_convex_combination(plant.p, plant.battery.c, plant.l)
            for hit_state, hit_prob in hit_state_list.items():
                J_expect[u_idx] += J[hit_state] * hit_prob
        J_update[i], Action_idx = torch.min(J_expect, dim=0)
        Actions[t][i] = int(Action_idx)
    J = J_update
print(get_pcl(int(torch.argmin(J))))
print(J)

p, c, l = 0, 0.5, 5
J_expect = torch.zeros(u_dis.shape[0])
u = -5
plant.reset(c, p, l)
plant.step(u, l)
hit_state_list = get_convex_combination(plant.p, plant.battery.c, plant.l)
for hit_state, hit_prob in hit_state_list.items():
    J_expect[6] += J[hit_state] * hit_prob
print(get_pcl(140))