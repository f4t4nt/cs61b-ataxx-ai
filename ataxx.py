import coax
import gym
import gym_ataxx
import haiku as hk
import jax
import jax.numpy as jnp
from optax import adam

# env with preprocessing
env = gym.make('ataxx-v0')  # AtariPreprocessing will do frame skipping
# env = gym.wrappers.AtariPreprocessing(env)
# env = coax.wrappers.FrameStacking(env, num_frames=3)
# env = coax.wrappers.TrainMonitor(env, name=name, tensorboard_dir=f"./data/tensorboard/{name}")


def func(S, is_training):
    """ type-2 q-function: s -> q(s,.) """
    seq = hk.Sequential((
        # coax.utils.diff_transform,
        # hk.Conv2D(16, kernel_shape = 8, stride = 4), jax.nn.relu,
        # hk.Conv2D(32, kernel_shape = 4, stride = 2), jax.nn.relu,
        # hk.Flatten(),
        # hk.Linear(256), jax.nn.relu,
        hk.Linear(env.action_space.n, w_init = jnp.zeros),
    ))
    X = jnp.stack(S, axis = -1) # stack frames
    return seq(X[0])


# function approximator
q = coax.Q(func, env)
pi = coax.EpsilonGreedy(q, epsilon = 1)

# target network
q_targ = q.copy()

# updater
qlearning = coax.td_learning.QLearning(q, q_targ = q_targ, optimizer=adam(3e-4))

# reward tracer and replay buffer
tracer = coax.reward_tracing.NStep(n = 1, gamma = 0.99)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity = 1000)


# DQN exploration schedule (stepwise linear annealing)
epsilon = coax.utils.StepwiseLinearFunction((0, 1), (1000, 0.1), (2000, 0.01))


win, lose, draw = 0, 0, 0
while env.T < 3000:
    prev_obs = env.reset(wall_p = 0.2)
    pi.epsilon = epsilon(env.T)

    while True:
        action = pi(prev_obs)
        obs, reward, done, info = env.step(action)

        # trace rewards and add transition to replay buffer
        tracer.add(prev_obs, action, reward, done)
        while tracer:
            buffer.add(tracer.pop())

        # learn
        if len(buffer) > 50000:  # buffer warm-up
            metrics = qlearning.update(buffer.sample(batch_size=32))
            env.record_metrics(metrics)

        if env.T % 10000 == 0:
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

# win, lose, draw = 0, 0, 0
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