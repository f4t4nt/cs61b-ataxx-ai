import gym
import gym_ataxx
env = gym.make('ataxx-v0')
w, l, d = 0, 0, 0
while True:
    prev = env.reset()
    # env.render()
    while True:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        # if prev != observation:
        #     print(action)
        #     env.render()
        #     x = 1
        prev = observation
        if info["winner"] == "white":
            w += 1
        elif info["winner"] == "black":
            l += 1
        elif info["winner"] == "draw":
            d += 1
        if done:
            break
    print('W', w, 'L', l, 'D', d)