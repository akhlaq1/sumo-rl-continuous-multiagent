import os

from maddpg.maddpg import MADDPG
from maddpg.replaybuffer import MultiAgentReplayBuffer
import traci
import numpy as np
import sys
from pathlib import Path
import json
from maddpg.csv_saver import CSV_Writer
from utils import plot_learning_curve
import sumo_env
import gymnasium as gym

os.environ['SUMO_LIBSUMO_FALLBACK'] = "1"

def progress(count, total, suffix=''):
    bar_len = 100
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()  # As suggested by Rom Ruben

def obs_list_to_state_vector(observation):
    '''
    This function flatens the array of observations
    in multiple agent the observation will be for each agent
    so this function flatens it
    '''

    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

def create_env():

    env = gym.make('sumo_env-v0',
                net_file='nets/RESCO/grid4x4/grid4x4.net.xml',
                route_file='nets/RESCO/grid4x4/grid4x4_1.rou.xml',
                out_csv_name='output.csv',
                use_gui=False,
                num_seconds=3600)
    observations, info = env.reset()

    return env, observations, info

# while env.agents:
#     actions = {agent: env.action_space(agent).sample() for agent in env.agents}  # this is where you would insert your policy
#     observations, rewards, terminations, truncations, infos = env.step(actions)

if __name__ == '__main__':
    env, observations, info = create_env()

    print("Observations")
    print(observations)
    print("===========================================")
    print("INFO")
    print(info)
    # print(env.observation_space(0))

    env.close()