"""SUMO Environment for Traffic Signal Control."""

from gymnasium.envs.registration import register


register(
    id="sumo_env-v0",
    entry_point="sumo_env.environment.env:SumoEnvironment",
    kwargs={"single_agent": False},
)
