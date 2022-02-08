import gfootball.env as football_env
from gfootball.env import wrappers


def create_env(env_id, left_agent, right_agent=0):
    env = football_env.create_environment(env_name=env_id, render=False, \
                                            representation='simple115v2', \
                                            number_of_left_players_agent_controls=left_agent, \
                                            number_of_right_players_agent_controls=right_agent)
    return env



if __name__ == '__main__':
    env = football_env.create_environment(env_name='11_vs_11_easy_stochastic', render=False, 
                                            representation='simple115v2',
                                            number_of_left_players_agent_controls=3,
                                            number_of_right_players_agent_controls=2)
    env.reset()

    done = False
    
    while not done:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
 