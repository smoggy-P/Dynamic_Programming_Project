import numpy as np
from LSTM import LSTM
from System import Plant
from Policy import Estimator
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

c_dis = np.arange(start=0, stop=8, step=1)
p_dis = np.arange(start=0, stop=18, step=1)
l_dis = np.arange(start=0, stop=11, step=1)
u_dis = np.arange(start=-5, stop=5, step=1)

n_sample = 100

n = c_dis.shape[0] * p_dis.shape[0] * l_dis.shape[0]
P_trans = np.zeros(shape=[n,n])

idx = 0
plant = Plant(dt=1/60)
estimator = Estimator()
std = 0.1 # variation of estimated load

for p in tqdm(p_dis):
    for c in c_dis:
        for l in l_dis:
            for u in u_dis:
                real_loads = np.random.normal(loc=estimator.estimate(l), scale=std, size=n_sample)
                for real_l in real_loads:
                    plant.reset(c, p, l)
                    plant.step(u, real_l)
            idx += 1

print(P_trans.shape)

