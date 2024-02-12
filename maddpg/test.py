import numpy as np

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


observation = [
    [1,2,3,4,5,6,7,8],
    [1,2,3,4,5,6,7,8,9,10],
    [1,2,3,4,5,6,7,8,9,10]
]


state = obs_list_to_state_vector(observation)

print(state)