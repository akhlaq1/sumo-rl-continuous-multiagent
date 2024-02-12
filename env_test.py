from custom_sumo_env.env import SumoEnvironemt
from maddpg.maddpg import MADDPG
from maddpg.replaybuffer import MultiAgentReplayBuffer
import traci
import numpy as np
import sys
from pathlib import Path
import json
from maddpg.csv_saver import CSV_Writer
from utils import plot_learning_curve


def progress(count, total, suffix=''):
    bar_len = 60
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


if __name__ == '__main__':
    env = SumoEnvironemt(episode_length=3000, delta_time=500)
    scenario = '4x4Grid_900vol'
    chkdir = './tmp/maddpg/'
    Path(chkdir + scenario).mkdir(parents=True, exist_ok=True)
    n_agents = env.n
    # actor_dims is the no. of observation passed to each agent
    actor_dims = [24, 24, 24, 24]
    critic_dims = sum(actor_dims)
    n_actions = env.action_space.shape[0]
    csvWriter = CSV_Writer(outfile="reward_waiting_time_neg")

    # action_header = [
    #     f"action_Agent{i}_Phase{k}" for k in range(n_actions)
    #     for i in range(n_agents)
    # ]

    observation_header = [
        f"obs_Agent{i}_{k}" for i in range(4)
        for k in [*["Density"]*12, *["Queue"]*12]
    ]
    # reward_header = [f"reward_Agent{i}" for i in range(n_agents)]

    csvWriter.header.append("episode")
    csvWriter.header.append('mean_100_episode_score')
    csvWriter.header.append('episode_score')
    csvWriter.header += observation_header
    # csvWriter.header += reward_header

    csvWriter.create_file()

    # print(
    #     f"n_agents {n_agents}, actor_dims {actor_dims}, critic_dims {critic_dims} n_actions {n_actions}"
    # )

    maddpg_agents = MADDPG(actor_dims,
                           critic_dims,
                           n_agents,
                           n_actions,
                           fc1=64,
                           fc2=64,
                           alpha=0.01,
                           beta=0.01,
                           gamma=0.95,
                           scenario=scenario,
                           chkpt_dir=chkdir,
                           action_space_high=env.action_space.high)

    memory = MultiAgentReplayBuffer(1000000,
                                    critic_dims,
                                    actor_dims,
                                    n_actions,
                                    n_agents,
                                    batch_size=1024)

    # CONFIG VARIABLES
    PRINT_INTERVAL = 1
    N_EPISOES = 10000
    N_Evaluation_EPISOES = 100
    MAX_STEPS = 25
    total_steps = 0
    score_history = []
    mean_100_score_history = []
    evaluate = False
    best_score = -40

    if evaluate:
        maddpg_agents.load_checkpoint()
        N_EPISOES = N_EPISOES

    for i in range(N_EPISOES):
        obs = env.reset()
        score = 0
        episode_score = []
        actions_csv = []
        obs_csv = []

        done = [False] * n_agents
        episode_step = 0

        print(f"New Episode Started No. {i}")
        while not any(done):
            # print(traci.simulation.getTime())
            progress(traci.simulation.getTime(), 3600)
            actions = maddpg_agents.choose_action(obs)

            # print(actions)
            obs_, reward, done, info = env.step(actions)

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            memory.store_transition(obs, state, actions, reward, obs_, state_,
                                    done)

            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            obs = obs_

            # score += sum(reward)
            episode_score.append(reward)
            total_steps += 1
            episode_step += 1

            # actionsList = np.round(obs_list_to_state_vector(actions), 2) * 100

            # actions_csv.append(actionsList.tolist())
            obs_csv.append(obs_list_to_state_vector(obs_))

            # data = [i, *actionsList.astype(int), *state_, *reward]
            # csvWriter.write_row(data)

        score_history.append(np.mean(episode_score))
        avg_score = np.mean(score_history[-100:])
        mean_100_score_history.append(avg_score)

        filename = 'Multi-ddpg_output.png'
        plot_learning_curve(mean_100_score_history,
                            filename, "pyTorch scores")

        obs_for_csv = np.mean(obs_csv, axis=0).tolist()

        data = [i, avg_score, np.mean(episode_score), *obs_for_csv]
        csvWriter.write_row(data)
        actions_csv = []
        obs_csv = []

        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score

        # if i % PRINT_INTERVAL == 0 and i > 0:
        print(
            f"Episode {i} Episode_Reward {round(score,2)} Average_Score_last_100 {round(avg_score,2)}"
        )
