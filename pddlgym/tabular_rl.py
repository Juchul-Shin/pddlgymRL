from collections import defaultdict
from random import randint
import random
import time
import numpy as np
import problem_generator
import pddlgym
import matplotlib.pyplot as plt
import warnings


DIR_LIST = ['north', 'east', 'south', 'west']
ACT_LIST = ['heading-forward', 'heading-left', 'heading-right']

class InvalidDirection(Exception):
    """See PDDLEnv docstring"""
    pass

# north를 기준으로 x,y 좌표 값을 회전 시킴
def rotate_coordinate(coordinate, direction, grid_size = 5):
    if (direction=='east'):
        t_x = grid_size - coordinate[1] + 1
        t_y = coordinate[0]
        return (t_x, t_y)
    elif (direction=='south'):
        t_x = grid_size - coordinate[0] + 1
        t_y = grid_size - coordinate[1] + 1
        return (t_x, t_y)
    elif (direction=='west'):
        t_x = coordinate[1]
        t_y = grid_size - coordinate[0] + 1
        return (t_x, t_y)
    elif (direction=='north'):
        return coordinate
    else:
        raise InvalidDirection(f"direction '{direction}' is invalid")


class TabularRL:
    def __init__(self, env, default_action_value = -10., grid_size=5, num_threat = 1, num_target = 0):
        self.env = env
        self.action_value = defaultdict(lambda : [default_action_value, default_action_value, default_action_value])
        self.action_space = env.action_space.predicates
        self.grid_size = grid_size
        self.num_threat = num_threat
        self.num_target = num_target
        self.target_acquired = False

    
    # PDDL Observation을 Grid의 Orient, Origin을 North, Drone 위치를 중심으로 좌표 변환하여 Tuple로 만들어 반환
    def _convert_obs(self, obs):
        threat_at = []
        transformed_threat = []
        for element in obs[0]:
            if (element.predicate.name == 'drone-at'):
                drone_at = element
            if (element.predicate.name == 'drone-to'):
                drone_to = element
            if (self.num_threat > 0):
                if (element.predicate.name == 'threat-at'):
                    threat_at.append(element)
            # if (element.predicate.name == 'target-at'):
            #     target_at = element
        drone_x = int(drone_at._str.split(':')[0].split('-')[2])
        drone_y = int(drone_at._str.split(':')[0].split('-')[3])
        drone_to = drone_to._str.split('(')[1].split(':')[0]
        #get goal,threat position
        goal_x = int(obs[2]._str.split(':')[0].split('-')[2])
        goal_y = int(obs[2]._str.split(':')[0].split('-')[3])
        #transform coordinate by direction        
        transformed_drone = rotate_coordinate((drone_x, drone_y), drone_to, self.grid_size)
        transformed_goal = rotate_coordinate((goal_x, goal_y), drone_to, self.grid_size)
        transformed_threat = []
        #transformed_target = ()

        for i in range(0, self.num_threat):
            threat_x = int(threat_at[i]._str.split(':')[0].split('-')[2])
            threat_y = int(threat_at[i]._str.split(':')[0].split('-')[3])
            transformed_threat.append((threat_x, threat_y))
        # if (self.num_threat == 1):
        #     threat_x = int(threat_at._str.split(':')[0].split('-')[2])
        #     threat_y = int(threat_at._str.split(':')[0].split('-')[3])
        #     transformed_threat = rotate_coordinate((threat_x, threat_y), drone_to, self.grid_size)
        # if (self.num_target > 0):
        #     target_x = int(target_at._str.split(':')[0].split('-')[2])
        #     target_y = int(target_at._str.split(':')[0].split('-')[3])
        #     transformed_target = rotate_coordinate((target_x, target_y), drone_to, self.grid_size)

        
        #get relative coordinate from drone to goal
   
        converted_obs = (*transformed_drone, *transformed_goal)

        for i in range(0, self.num_threat):
            converted_obs = (*converted_obs, *transformed_threat[i])
        # if (self.num_target > 0):
        #     if (not self.target_acquired):
        #         if (transformed_target[0] == transformed_drone[0]) and (transformed_target[1] == transformed_drone[1]):
        #             self.target_acquired = True
        #     converted_obs = (*converted_obs, *transformed_target, self.target_acquired)
        
        return converted_obs

    def sample_action(self):
        valid_actions = list(sorted(self.env.action_space.all_ground_literals(self.obs, valid_only=True)))
        random_action = valid_actions[randint(0, len(valid_actions) - 1)]
        return random_action

    # 현재 Q-Value에 대한 Greedy Policy로 에피소드를 num_episodes 마늠 수행
    def run_episodes(self, env, num_episodes, to_print = False):
        total_rewards = []
        for e in range(num_episodes):
            obs, _ = env.reset()
            tabular_state = self._convert_obs(obs)
            done = False
            game_reward = 0
            num_steps = 0      #Step 수를 제한
            self.target_acquired = False
            first_acquired = False
            while not done:
                # select a greedy action
                valid_actions = list(sorted(env.action_space.all_ground_literals(obs, valid_only=True)))
                action, _ = self.greedy(tabular_state, valid_actions)
                if to_print:
                    self.printobs(obs)
                    print(action.predicate.name)
                next_obs, rew, done, _ = env.step(action)
                obs = next_obs
                tabular_state = self._convert_obs(obs)

                if self.num_target > 0:
                    if ((not done) and (self.target_acquired) and (not first_acquired)):
                        first_acquired = True
                        reward = 10
                    elif (done and self.target_acquired):
                        reward = 20

                game_reward += rew 
                
                if done or num_steps > 50:
                    total_rewards.append(game_reward)
                    if to_print and done and self.target_acquired:
                        print("Agent has arrived at the goal position acquiring the target successfully\n\n")
                    elif to_print and done:
                        print("Agent has arrived at the goal position\n\n")
                    elif to_print:
                        print("Agent couldn't arrive at ")
                    break
                num_steps += 1

        if to_print:
            print('Mean score: %.3f of %i games!'%(np.mean(total_rewards), num_episodes))

        return np.mean(total_rewards)

        # if (not policy):
        #     while (True):
        #         print('Position({0},{1}) Direction {2}'.format(self.drone_x, self.drone_y, self.drone_to))
        #         action = self.sample_action()
        #         print(action)
        #         self.obs, reward, done, _ = self.env.step(action)
        #         self.state = self._convert_obs(self.obs)
                
        #         if (done):
        #             print('----------Episode End-------------\n')
        #             break

    
    #### valid actions이 아닌 index가 뽑힌 경우 no greedy action에 빠짐
    # def greedy(self, state, valid_actions):
    #     if (state in self.action_value):
    #         action = valid_actions[randint(0, len(valid_actions) - 1)]
    #         for index, action_name in enumerate(ACT_LIST):
    #             if action_name == action.predicate.name:
    #                 return action, index
    #     q_values = np.array(self.action_value[state])
    #     # state에서 valid action이 아닌 경우는 큰 음의 값을 설정
    #     valid_action_list = [action.predicate.name for action in valid_actions]
    #     for index, action in enumerate(ACT_LIST):
    #         if not action in valid_action_list:
    #             q_values[index] = np.NINF
    #             self.action_value[state][index] = np.NINF
        
    #     #Q값이 가장 큰 valid action 선택
    #     while(len(q_values) > 0):
    #         index = np.argmax(q_values)
    #         for action in valid_actions:
    #             if (action.predicate.name == ACT_LIST[index]):
    #                 return action, index
    #         q_values = np.delete(q_values, index)
        
    #     print("****** There is no greedy action *****")
    #     print(self.env)
    #     return None, None

    #### valid actions이 아닌 index가 뽑힌 경우 no greedy action에 빠짐
    def greedy(self, state, valid_actions):
        q_values = []
        if (not (state in self.action_value)): # 처음 방문하는 state는 invalid action 처리 후 랜덤 액션 반환
            # state에서 valid action이 아닌 경우는 큰 음의 값을 설정
            q_values = np.array(self.action_value[state])
            valid_action_list = [action.predicate.name for action in valid_actions]
            for index, action in enumerate(ACT_LIST):
                if not action in valid_action_list:
                    q_values[index] = np.NINF
                    self.action_value[state][index] = np.NINF
            action = random.choice(valid_actions)
            index = ACT_LIST.index(action.predicate.name)
            return action, index
        
        # 기존 방문 state에서는 #Q값이 가장 큰 valid action 선택
        q_values = np.array(self.action_value[state])
        sorted_indices = np.argsort(-q_values)

        for i in range(0,3):
            index = sorted_indices[i]
            for action in valid_actions:
                if (action.predicate.name == ACT_LIST[index]):
                    return action, index

        print("****** There is no greedy action *****")
        print(self.env)
        return None, None

    def is_aquired_target(self, next_tabular_state):
        return (next_tabular_state[4] == 0) and (next_tabular_state[5] == 0)


    def e_greedy(self, state, epsilon, valid_actions):
        '''
        Epsilon greedy policy
        '''
        if np.random.uniform(0,1) < epsilon:
            # Choose a random action
            action = random.choice(valid_actions)
            for index, action_name in enumerate(ACT_LIST):
                if action_name == action.predicate.name:
                    return action, index
        else:
            # Choose the action of a greedy policy
            return self.greedy(state, valid_actions)
        
    # 현재 observation 중 grid와 관련된 내용 출력
    def printobs(self, obs):
        threat_at = []
        for element in obs[0]:
            if (element.predicate.name == 'drone-at'):
                drone_at = element
            if (element.predicate.name == 'drone-to'):
                drone_to = element
            if (element.predicate.name == 'threat-at'):
                threat_at.append(element)
            if (element.predicate.name == 'target-at'):
                target_at = element
        
        drone_x = int(drone_at._str.split(':')[0].split('-')[2])
        drone_y = int(drone_at._str.split(':')[0].split('-')[3])
        drone_to = drone_to._str.split('(')[1].split(':')[0]
        obs_str = 'Drone-At[{0},{1}], Drone-To[{2}]'.format(drone_x, drone_y, drone_to)
        #get goal position
        goal_x = int(obs[2]._str.split(':')[0].split('-')[2])
        goal_y = int(obs[2]._str.split(':')[0].split('-')[3])
        obs_str += ', Goal-At[{},{}]'.format(goal_x, goal_y)
        for i in range(0, self.num_threat):
            threat_x = int(threat_at[i]._str.split(':')[0].split('-')[2])
            threat_y = int(threat_at[i]._str.split(':')[0].split('-')[3])
            obs_str += ', Threat-At[{},{}]'.format(threat_x, threat_y)         

        if self.num_target > 0:
            target_x = int(target_at._str.split(':')[0].split('-')[2])
            target_y = int(target_at._str.split(':')[0].split('-')[3])
            obs_str += ', Target-At[{},{}]'.format(target_x, target_y)        
        
        print(obs_str)


    def Q_learning(self, alpha = 0.02, num_episodes=5001, epsilon=0.3, gamma = 0.9, decay = 0.0001):
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
        first_acquired = False
        for ep in range(1, num_episodes):
            obs , _ = self.env.reset()
            tabular_state = self._convert_obs(obs)
            done = False
            total_reward = 0
            self.target_acquired = False
            step_count = 0

            if epsilon > 0.001:# and ep > 10000 :
                epsilon -= decay
            
            while not done:
                step_count += 1
                valid_actions = list(sorted(self.env.action_space.all_ground_literals(obs, valid_only=True)))
                if (len(valid_actions)== 0):
                    print("Running pisode is stuck, a new episode start")
                    done == True
                    break
                action, index = self.e_greedy(tabular_state, epsilon, valid_actions)
                
                if action is None:
                    print("Running pisode is stuck, a new episode start")
                    done == True
                    break
                
                next_obs, reward, done, _ = self.env.step(action)
                          
                next_tabular_state  = self._convert_obs(next_obs)

                # Target 획득이 필요한 Problem에 대한 Reward
                if self.num_target > 0:
                    # if ((not done) and (self.target_acquired) and (not first_acquired)):
                    #     first_acquired = True
                    #     reward = 10
                    # elif (done and self.target_acquired):
                    #     reward = 20
                    if (done and not self.target_acquired):
                        done = False

                next_max_q = np.max(np.array(self.action_value[next_tabular_state]))
                if (next_max_q != np.NINF) and (self.action_value[tabular_state][index] != np.NINF):
                    try:
                        self.action_value[tabular_state][index] = self.action_value[tabular_state][index]\
                                                           + alpha * (reward + gamma * next_max_q\
                                                                      - self.action_value[tabular_state][index])
                    except RuntimeWarning as e:
                        print("Warning Raised : ", str(e))
                else:
                    a = self.action_value[tabular_state][index]
                    print(a)

                obs = next_obs
                tabular_state = next_tabular_state
                total_reward += reward

                if (done):
                    games_reward.append(total_reward)
            total_steps += step_count
            if (ep % 100) == 0:
                test_reward = self.run_episodes(test_env, 30)
                print("Learning Test Episode:{:5d}  Eps:{:2.4f}  Rew:{:2.4f}".format(ep, epsilon, test_reward))
                print("Total Visited States : {}", len(self.action_value))
                print("Total Steps : {}", total_steps)
                test_rewards.append(test_reward)
                episodes.append(ep)
                total_visited_states.append(len(self.action_value))

        fig, ax1 = plt.subplots()
        ax1.plot(episodes, test_rewards, 'b-')  # 파란색 실선으로 그래프 그리기
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('AVG. Reward', color='b')
        ax1.tick_params('y', colors='b', labelleft=True, labelright=False)
        ax2 = ax1.twinx()
        ax2.plot(episodes, total_visited_states, 'r-')
        ax2.set_ylabel('Total Visited States', color='r')
        ax2.tick_params('y', colors='r', labelleft=False, labelright=True)
        plt.title('Q-Learning AVG. Reward & Visited States per 100 Episodes\nalpha={0}, initial_epsilon={1}, gamma = {2}, decay={3}'.format(alpha, initial_epsilon, gamma, decay))  # 그래프 제목 설정
        plt.show()  # 그래프 출력1

    def SARSA(self, alpha = 0.02, num_episodes=5001, epsilon=0.3, gamma = 0.9, decay = 0.0001):
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

        for ep in range(1, num_episodes):
            obs , _ = self.env.reset()
            tabular_state = self._convert_obs(obs)
            done = False
            total_reward = 0
            self.target_acquired = False
            step_count = 0
            first_acquired = False
            # decay the epsilon value until it reaches the threshold of 0.01
            if epsilon > 0.001:# and ep > 50000:
                epsilon -= decay

            valid_actions = list(sorted(self.env.action_space.all_ground_literals(obs, valid_only=True)))
            if (len(valid_actions)== 0):
                print("Running pisode is stuck, a new episode start")
                done == True
                break
            action, action_index = self.e_greedy(tabular_state, epsilon, valid_actions)

            # loop the main body until the environment stops
            while not done:
                step_count += 1

                
                if action is None:
                    print("Running pisode is stuck, a new episode start")
                    done == True
                    break
                
                next_obs, reward, done, _ = self.env.step(action)             
                next_tabular_state  = self._convert_obs(next_obs)
                valid_actions = list(sorted(self.env.action_space.all_ground_literals(next_obs, valid_only=True)))
                next_action, next_action_index = self.e_greedy(next_tabular_state, epsilon, valid_actions)
                # Target 획득이 필요한 Problem에 대한 Reward
                if self.num_target > 0:
                    # if ((not done) and (self.target_acquired) and (not first_acquired)):
                    #     first_acquired = True
                    #     reward = 10
                    # elif (done and self.target_acquired):
                    #     reward = 20
                    if (done and not self.target_acquired):
                        done = False
                self.action_value[tabular_state][action_index] = self.action_value[tabular_state][action_index]\
                                                + alpha * (reward + gamma * self.action_value[next_tabular_state][next_action_index]\
                                                - self.action_value[tabular_state][action_index])

                obs = next_obs
                tabular_state = next_tabular_state
                action = next_action
                action_index = next_action_index
                total_reward += reward

                if (done):
                    games_reward.append(total_reward)

            total_steps += step_count
            if (ep % 100) == 0:
                test_reward = self.run_episodes(test_env, 30)
                print("Learning Test Episode:{:5d}  Eps:{:2.4f}  Rew:{:2.4f}".format(ep, epsilon, test_reward))
                print("Total Visited States : {}", len(self.action_value))
                print("Total Steps : {}", total_steps)
                test_rewards.append(test_reward)
                episodes.append(ep)
                total_visited_states.append(len(self.action_value))

        fig, ax1 = plt.subplots()
        ax1.plot(episodes, test_rewards, 'b-')  # 파란색 실선으로 그래프 그리기
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('AVG. Reward', color='b')
        ax1.tick_params('y', colors='b', labelleft=True, labelright=False)
        ax2 = ax1.twinx()
        ax2.plot(episodes, total_visited_states, 'r-')
        ax2.set_ylabel('Total Visited States', color='r')
        ax2.tick_params('y', colors='r', labelleft=False, labelright=True)
        plt.title('SARSA AVG. Reward & Visited States per 100 Episodes\nalpha={0}, initial_epsilon={1}, gamma = {2}, decay={3}'.format(alpha, initial_epsilon, gamma, decay))  # 그래프 제목 설정
        plt.show()  # 그래프 출력1


    def Test(self):
        env = pddlgym.make("PDDLEnvDroneTest-v0")
        test_reward = self.run_episodes(self.env, 30, True)
        print("Total Reward of 10 tests:{0}".format(test_reward))

    def __str__(self):
        str(self.state)
        
if __name__ == '__main__':

    env = pddlgym.make("PDDLEnvDrone-v0", seed = int(time.time()))
    rl = TabularRL(env, num_target=0, num_threat=2)
    rl.Q_learning()
    #rl.SARSA()
    rl.Test()