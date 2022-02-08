import gym
import random
from threading import Thread
from msagent.envs.fighting.env.fighting_env_twoplayers import play_thread
import time

from msagent.envs.fighting.env.gym_ai import GymAI

def p_thread1(env,p1,p2):
    while True:
        obs = env.reset(p1,p2)

        done = False
        while not done:
            new_obs, reward, done, _ = env.step(random.randint(0, 55))

def p_thread2(env,p1,p2):
    while True:
        obs = env.reset(p1,p2)

        done = False
        while not done:
            new_obs, reward, done, _ = env.step(random.randint(0, 55))

env1 = gym.make("FightingiceDataTwoPlayer-v0", java_env_path="",port=4242, freq_restart_java=3)
p2_server = env1.build_pipe_and_return_p2()
env2 = gym.make("FightingiceDataTwoPlayer-v0", java_env_path="",port=4242,p2_server=p2_server)
t1 = Thread(target=p_thread1, name="play_thread1", args=(env1,GymAI,GymAI, ))
t2 = Thread(target=p_thread2, name="play_thread2", args=(env2,GymAI,GymAI, ))
t1.start()
time.sleep(10)
t2.start()
while True:
    t1.join()
    t2.join()

print("finish")
