import gym
import gym_ataxx

env = gym.make('ataxx-v0')
win, lose, draw = 0, 0, 0
while True:
    prev = env.reset(wall_p = 0.2)
    env.render()
    while True:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if prev != observation:
            env.render()
        prev = observation
        if info["winner"] == "white":
            win += 1
        elif info["winner"] == "black":
            lose += 1
        elif info["winner"] == "draw":
            draw += 1
        if done:
            break
    # print('W', win, 'L', lose, 'D', draw)