from collections import defaultdict
from random import randint
import random
import time
import numpy as np
import pddlgym
import matplotlib.pyplot as plt
import os
from gym.envs.registration import register
import pickle
from PIL import Image
import json

DIR_LIST = ['north', 'east', 'south', 'west']
ACT_LIST = ['heading-forward', 'heading-left', 'heading-right']

class InvalidDirection(Exception):
    """See PDDLEnv docstring"""
    pass




class TabularRL:
    def __init__(self, env, default_action_value = 0., grid_size=5, num_threat = 1, num_target = 0, save = True):
        self.env = env
        self.action_value = defaultdict(lambda : [default_action_value, default_action_value, default_action_value])
        self.save = save
        #self.grid_size = grid_size
        #self.num_threat = num_threat
        #self.num_target = num_target
        #self.target_acquired = False

    def sample_action(self):
        valid_actions = list(sorted(self.env.action_space.all_ground_literals(self.obs, valid_only=True)))
        random_action = valid_actions[randint(0, len(valid_actions) - 1)]
        return random_action

    def greedy(self, state):
        # 기존 방문 state에서는 #Q값이 가장 큰 valid action 선택
        q_values = []
        index = -1
        
        if state in self.action_value:
             # 방문한적이 있는 state에서는 greedy하게 선택
            q_values = np.array(self.action_value[state])
            if np.allclose(q_values, q_values[0]):
                index = np.random.randint(len(q_values))
            else:
                index = np.argmax(q_values)
       
        else:
            # 처음 방문한 state에서는 random하게 액션 선택
            q_values = np.array(self.action_value[state])
            index = np.random.randint(len(q_values))

        return ACT_LIST[index], index

    def e_greedy(self, state, epsilon):
        if np.random.uniform(0,1) < epsilon:
            # Choose a random action
            index = np.random.randint(len(ACT_LIST))

            return ACT_LIST[index], index
        else:
            # Choose the action of a greedy policy
            return self.greedy(state)
        
    # 현재 Q-Value에 대한 Greedy Policy로 에피소드를 num_episodes 마늠 수행
    def run_episodes(self, env, num_episodes, to_print = False, render = False):
        total_rewards = []
        frames = []
        for e in range(num_episodes):
            obs, _ = env.reset()
            if (render):
                frame = env.render()
                frames.append(frame)            
            done = False
            game_reward = 0
            num_steps = 0      #Step 수를 제한
            while not done:
                # select a greedy action
                action, _ = self.greedy(obs)
                if to_print:
                    env.print_obs()
                    print(action)
                next_obs, rew, done, _ = env.step(action)
                if (render):
                    frame = env.render()
                    frames.append(frame)                  
                obs = next_obs
                game_reward += rew 
                
                if done or num_steps > 50:
                    total_rewards.append(-num_steps)
                    #total_rewards.append(game_reward)
                    # if to_print and done and self.target_acquired:
                    #     print("Agent has arrived at the goal position acquiring the target successfully\n\n")
                    # elif to_print and done:
                    #     print("Agent has arrived at the goal position\n\n")
                    if to_print:
                        print("Agent arrived at the goal\n")
                    break
                num_steps += 1

        if to_print:
            print('Mean score: %.3f of %i games!'%(np.mean(total_rewards), num_episodes))

        # GIF 파일로 저장
        if (render):
            frames[0].save('animation.gif', format='GIF', append_images=frames[1:], save_all=True, duration=1000, loop=0)

        return np.mean(total_rewards)


        
    # 현재 observation 중 grid와 관련된 내용 출력



    def Q_learning(self, alpha = 0.01, num_episodes=30000, epsilon=0.3, gamma = 0.90, decay = 0.00001, num_threat = 0):
        
        games_reward = []
        test_rewards = []
        episodes = []
        initial_epsilon = epsilon
        test_env = pddlgym.make("PDDLEnvDroneTest-v0", seed = int(time.time()))

        test_reward = self.run_episodes(test_env, 30)
        test_rewards.append(test_reward)
        episodes.append(0)
        total_steps = 0
        total_visited_states = []
        total_visited_states.append(0)
        for ep in range(1, num_episodes+1):
            obs , _ = self.env.reset()
            done = False
            total_reward = 0
            step_count = 0

            if epsilon > 0.001:
                epsilon -= decay
            
            while not done:
                step_count += 1
                action, index = self.e_greedy(obs, epsilon)
                next_obs, reward, done, _ = self.env.step(action)
                          
                next_max_q = np.max(np.array(self.action_value[next_obs]))
                self.action_value[obs][index] = self.action_value[obs][index]\
                                                + alpha * (reward + gamma * next_max_q\
                                                            - self.action_value[obs][index])
 
                obs = next_obs
                total_reward += reward

                if (done):
                    games_reward.append(total_reward)

            total_steps += step_count
            if (ep % 100) == 0:
                test_reward = self.run_episodes(test_env, 100)
                print("Learning Test Episode:{:5d}  Eps:{:2.4f}  Rew:{:2.4f}".format(ep, epsilon, test_reward))
                print("Total Visited States : {}", len(self.action_value))
                print("Total Steps : {}", total_steps)
                test_rewards.append(test_reward)
                episodes.append(ep)
                total_visited_states.append(len(self.action_value))

        # 파일로 저장

        if self.save:
            policy = dict(self.action_value)
            # dictionary를 파일로 저장
            with open('ql.pol', 'wb') as file:
                pickle.dump(policy, file)

        fig, ax1 = plt.subplots()
        ax1.plot(episodes, test_rewards, 'b-')  # 파란색 실선으로 그래프 그리기
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('AVG. Stpes', color='b')
        ax1.tick_params('y', colors='b', labelleft=True, labelright=False)
        ax2 = ax1.twinx()
        ax2.plot(episodes, total_visited_states, 'r-')
        ax2.set_ylabel('Total Visited States', color='r')
        ax2.tick_params('y', colors='r', labelleft=False, labelright=True)
        plt.title('Q-Learning {4}-Threat Grid \nAVG. Steps & Visited States per 100 Episodes\nalpha={0}, initial_epsilon={1}, gamma = {2}, decay={3}\nReward 1(arrival), -0.001(step)'.format(alpha, initial_epsilon, gamma, decay,num_threat))  # 그래프 제목 설정
        plt.show(block=False)  # 그래프 출력1

        return self.action_value


    def SARSA(self, alpha = 0.02, num_episodes=2000, epsilon=0.3, gamma = 0.9, decay = 0.0001, num_threat = 0):
        games_reward = []
        test_rewards = []
        episodes = []
        initial_epsilon = epsilon
        test_env = pddlgym.make("PDDLEnvDroneTest-v0", seed = int(time.time()))

        test_reward = self.run_episodes(test_env, 30)
        test_rewards.append(test_reward)
        episodes.append(0)
        total_steps = 0
        total_visited_states = []
        total_visited_states.append(0)

        for ep in range(1, num_episodes+1):
            obs , _ = self.env.reset()
            done = False
            total_reward = 0
            step_count = 0
            # decay the epsilon value until it reaches the threshold of 0.01
            if epsilon > 0.001:# and ep > 50000:
                epsilon -= decay

            action, action_index = self.e_greedy(obs, epsilon)

            # loop the main body until the environment stops
            while not done:
                step_count += 1

                next_obs, reward, done, _ = self.env.step(action)  
                next_action, next_action_index = self.e_greedy(next_obs, epsilon)

                self.action_value[obs][action_index] = self.action_value[obs][action_index]\
                                                + alpha * (reward + gamma * self.action_value[next_obs][next_action_index]\
                                                - self.action_value[obs][action_index])

                obs = next_obs
                action = next_action
                action_index = next_action_index
                total_reward += reward

                if (done):
                    games_reward.append(total_reward)

            total_steps += step_count
            if (ep % 100) == 0:
                test_reward = self.run_episodes(test_env, 100)
                print("Learning Test Episode:{:5d}  Eps:{:2.4f}  Rew:{:2.4f}".format(ep, epsilon, test_reward))
                print("Total Visited States : {}", len(self.action_value))
                print("Total Steps : {}", total_steps)
                test_rewards.append(test_reward)
                episodes.append(ep)
                total_visited_states.append(len(self.action_value))

        if self.save:
            policy = dict(self.action_value)
            # dictionary를 파일로 저장
            with open('salsa.pol', 'wb') as file:
                pickle.dump(policy, file)

        fig, ax1 = plt.subplots()
        ax1.plot(episodes, test_rewards, 'b-')  # 파란색 실선으로 그래프 그리기
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('AVG. Steps', color='b')
        ax1.tick_params('y', colors='b', labelleft=True, labelright=False)
        ax2 = ax1.twinx()
        ax2.plot(episodes, total_visited_states, 'r-')
        ax2.set_ylabel('Total Visited States', color='r')
        ax2.tick_params('y', colors='r', labelleft=False, labelright=True)
        plt.title('SARSA {4}-Threat Grid \nAVG. Steps & Visited States per 100 Episodes\nalpha={0}, initial_epsilon={1}, gamma = {2}, decay={3}\nReward 1(arrival), -0.001(step)'.format(alpha, initial_epsilon, gamma, decay,num_threat))  # 그래프 제목 설정
        plt.show(block=False)  # 그래프 출력1


    def Test(self):
        env = pddlgym.make("PDDLEnvDroneTest-v0")
        test_reward = self.run_episodes(self.env, 1, True, True)
        print("Total Reward of 1 tests:{0}".format(test_reward))

    def __str__(self):
        str(self.state)

    def load_policy(self, file_name):
        # 파일에서 로드
        with open(file_name, 'rb') as file:
            policy = pickle.load(file)
            self.action_value.update(policy)


def register_drone_env(name, num_threat, is_test_env = False):
    name = "drone"
    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pddl")
    domain_file = os.path.join(dir_path, "drone.pddl")
    gym_name = name.capitalize()
    problem_dirname = name.lower()
    if is_test_env:
        gym_name += 'Test'
        problem_dirname += '_test'
    problem_dir = os.path.join(dir_path, problem_dirname)
    args = {'operators_as_actions' : True,
            'dynamic_action_space' : True,
            "raise_error_on_invalid_action": False,
            "num_threat": num_threat}
    register(
        id='PDDLEnv{}-v0'.format(gym_name),
        entry_point='drone_env:DroneEnv',
        kwargs=dict({'domain_file' : domain_file, 'problem_dir' : problem_dir,
                     **args}),
    )

if __name__ == '__main__':
    threats = 2
    save = True
    register_drone_env("drone", threats)
    register_drone_env("drone", threats, is_test_env=True)

    env = pddlgym.make("PDDLEnvDrone-v0", seed = int(time.time()))
    rl = TabularRL(env, num_target=0, num_threat=threats)
    
    rl.Q_learning(num_threat=threats)
    #rl.SARSA(num_threat=threats)
    #rl.load_policy('ql.pol')

    rl.Test()