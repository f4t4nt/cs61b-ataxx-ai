import coax
import gym
import gym_ataxx
import haiku as hk
import jax
import jax.numpy as jnp
from optax import adam

env = gym.make('ataxx-v0')

def func(S, is_training):
    seq = hk.Sequential((
        hk.Linear(env.action_space.n, w_init = jnp.zeros),
    ))
    X = jnp.stack(S, axis = -1)
    return seq(X[0])

q = coax.Q(func, env)
q.params = coax.utils.load('model.dmp')
q._params = q.params
pi = coax.EpsilonGreedy(q, epsilon = 1)
q_targ = q.copy()
qlearning = coax.td_learning.QLearning(q, q_targ = q_targ, optimizer=adam(3e-4))
tracer = coax.reward_tracing.NStep(n = 1, gamma = 0.99)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity = 100000)
epsilon = coax.utils.StepwiseLinearFunction((0, 1), (1000, 0.1), (2000, 0.01))
win, lose, draw = 0, 0, 0

while env.T < 100:
    prev_obs = env.reset(wall_p = 0.2)
    pi.epsilon = epsilon(env.T)
    while True:
        action = pi(prev_obs)
        obs, reward, done, info = env.step(action)
        tracer.add(prev_obs, action, reward, done)
        while tracer:
            buffer.add(tracer.pop())
        if len(buffer) > 32:
            metrics = qlearning.update(buffer.sample(batch_size=1))
        if env.T % 10 == 0:
            q_targ.soft_update(q, tau = 1)
        if not jnp.array_equal(prev_obs, obs):
            env.render()
        if done:
            if info["winner"] == "white":
                win += 1
            elif info["winner"] == "black":
                lose += 1
            elif info["winner"] == "draw":
                draw += 1
            break
        prev_obs = obs
    print('W', win, 'L', lose, 'D', draw)

# while True:
#     prev_obs = env.reset(wall_p = 0.2)
#     # env.render()
#     while True:
#         action = env.action_space.sample()
#         obs, reward, done, info = env.step(action)
#         # if prev_obs != obs:
#         #     env.render()
#         prev_obs = obs
#         if info["winner"] == "white":
#             win += 1
#         elif info["winner"] == "black":
#             lose += 1
#         elif info["winner"] == "draw":
#             draw += 1
#         if done:
#             break
#     print('W', win, 'L', lose, 'D', draw)