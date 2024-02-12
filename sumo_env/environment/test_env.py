from env import SumoEnvironment

env = SumoEnvironment(
    net_file='/home/ady/sumorl_maddpg_custom_env_v2/nets/RESCO/grid4x4/grid4x4.net.xml',
    route_file='/home/ady/sumorl_maddpg_custom_env_v2/nets/RESCO/grid4x4/grid4x4_1.rou.xml',
    out_csv_name='output.csv',
    use_gui=False,
    num_seconds=3600
)

print(env)

env.close()