from gym.envs.registration import register

register(
    id = 'ataxx-v0',
    entry_point = 'gym_ataxx.envs:AtaxxEnv',
)