import argparse

def get_args():
    parser = argparse.ArgumentParser(description='DP Arguments')
    parser.add_argument('--c_resolution',
                        type=float,
                        default=1.0,
                        help='c_resolution')
    parser.add_argument('--p_resolation',
                        type=float,
                        default=1.0,
                        help='p_resolation')
    parser.add_argument('--l_resolution',
                        type=float,
                        default=1.0,
                        help='l_resolution')
    parser.add_argument('--u_resolution',
                        type=float,
                        default=1.0,
                        help='u_resolution')
    
    parser.add_argument('--dt',
                        type=int,
                        default=2,
                        help='dt (min)')
    
    parser.add_argument('--T',
                        type=int,
                        default=360,
                        help='DP time steps')
    
    parser.add_argument('--n_sample',
                        type=int,
                        default=50,
                        help='monte carlo samples')
    
    parser.add_argument('--std',
                        type=float,
                        default=0.1,
                        help='std of estimated load')
    
    parser.add_argument('--save_action_table',
                        type=bool,
                        default=True,
                        help='std of estimated load')
    
    return parser.parse_args()