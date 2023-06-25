from collections import defaultdict
from random import randint
import random
import time
import numpy as np
import problem_generator
import pddlgym
import matplotlib.pyplot as plt


DIR_LIST = ['north', 'east', 'south', 'west']
ACT_LIST = ['heading-forward', 'heading-left', 'heading-right']

class InvalidDirection(Exception):
    """See PDDLEnv docstring"""
    pass

def transform_coordinate(coordinate, direction, grid_size = 5):
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

def relative_coordinate(drone, other, grid_size=5):
        return (other[0]-drone[0], other[1]-drone[1])



class TabularRL:
    def __init__(self, env, default_value = -10., grid_size=5):
        self.env = env
        #self.state_value = defaultdict(lambda : [default_value, default_value, default_value])
        self.action_value = defaultdict(lambda : [default_value, default_value, default_value])
        self.action_space = env.action_space.predicates
        self.grid_size = grid_size
    
    def _convert_obs(self, obs):
        for element in obs[0]:
            if (element.predicate.name == 'drone-at'):
                drone_at = element
            if (element.predicate.name == 'drone-to'):
                drone_to = element
        
        drone_x = int(drone_at._str.split(':')[0].split('-')[2])
        drone_y = int(drone_at._str.split(':')[0].split('-')[3])
        drone_to = drone_to._str.split('(')[1].split(':')[0]
        #get goal position
        goal_x = int(obs[2]._str.split(':')[0].split('-')[2])
        goal_y = int(obs[2]._str.split(':')[0].split('-')[3])
        #transform coordinate by direction        
        transformed_drone = transform_coordinate((drone_x, drone_y), drone_to, self.grid_size)
        transformed_goal = transform_coordinate((goal_x, goal_y), drone_to, self.grid_size)
        #get relative coordinate from drone to goal
        relative_pos_obs =  (transformed_drone[0], transformed_drone[1], \
                                 transformed_goal[0], transformed_goal[1])

        return relative_pos_obs

    def sample_action(self):
        valid_actions = list(sorted(self.env.action_space.all_ground_literals(self.obs, valid_only=True)))
        random_action = valid_actions[randint(0, len(valid_actions) - 1)]
        return random_action

    def run_episodes(self, env, num_episodes, to_print = False):
        '''
        Run some episodes to test the policy
        '''
        total_rewards = []
        for e in range(num_episodes):
            obs, _ = env.reset()
            tabular_state = self._convert_obs(obs)
            done = False
            game_reward = 0
            step_limit = 0

            while not done:
                # select a greedy action


                valid_actions = list(sorted(env.action_space.all_ground_literals(obs, valid_only=True)))
                action, _ = self.greedy(tabular_state, valid_actions)

                if to_print:
                    self.printobs(obs)
                    print(action)
                next_obs, rew, done, _ = env.step(action)
                obs = next_obs
                tabular_state = self._convert_obs(obs)
                game_reward += rew 
                
                if done or step_limit > 50:
                    total_rewards.append(game_reward)
                    if to_print:
                        print("Agent has arrived at the goal position\n\n")
                    break
                step_limit += 1

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
    def greedy(self, state, valid_actions):
        q_values = np.array(self.action_value[state])
        # state에서 valid action이 아닌 경우는 큰 음의 값을 설정
        valid_action_list = [action.predicate.name for action in valid_actions]
        for index, action in enumerate(ACT_LIST):
            if not action in valid_action_list:
                q_values[index] = np.NINF
                self.action_value[state][index] = np.NINF
        
        #Q값이 가장 큰 valid action 선택
        while(len(q_values) > 0):
            index = np.argmax(q_values)
            for action in valid_actions:
                if (action.predicate.name == ACT_LIST[index]):
                    return action, index
            q_values = np.delete(q_values, index)
        
        print("****** There is no greedy action *****")
        print(self.env)
        return None, None
        

    def e_greedy(self, state, epsilon, valid_actions):
        '''
        Epsilon greedy policy
        '''
        if np.random.uniform(0,1) < epsilon:
            # Choose a random action
            action = valid_actions[randint(0, len(valid_actions) - 1)]
            for index, action_name in enumerate(ACT_LIST):
                if action_name == action.predicate.name:
                    return action, index
        else:
            # Choose the action of a greedy policy
            return self.greedy(state, valid_actions)
        
    def printobs(self, obs):
        for element in obs[0]:
            if (element.predicate.name == 'drone-at'):
                drone_at = element
            if (element.predicate.name == 'drone-to'):
                drone_to = element

        drone_x = int(drone_at._str.split(':')[0].split('-')[2])
        drone_y = int(drone_at._str.split(':')[0].split('-')[3])
        drone_to = drone_to._str.split('(')[1].split(':')[0]
        #get goal position
        goal_x = int(obs[2]._str.split(':')[0].split('-')[2])
        goal_y = int(obs[2]._str.split(':')[0].split('-')[3])
        #transform coordinate by direction        
        print('Drone-At[{0},{1}], Drone-To[{2}], Goal-At[{3},{4}]'.format(drone_x, drone_y, drone_to, goal_x, goal_y))


    def Q_learning(self, alpha = 0.03, num_episodes=3001, epsilon=0.3, gamma = 0.95, decay = 0.0001):
        games_reward = []
        test_rewards = []
        episodes = []
        initial_epsilon = epsilon
        for ep in range(1, num_episodes):
            obs , _ = self.env.reset()
            tabular_state = self._convert_obs(obs)
            done = False
            total_reward = 0


            if epsilon > 0.001 :
                epsilon -= decay
            
            while not done:
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
                
                next_tarbular_state = self._convert_obs(next_obs)
                next_max_q = np.max(np.array(self.action_value[next_tarbular_state]))
                self.action_value[tabular_state][index] = self.action_value[tabular_state][index]\
                                                           + alpha * (reward + gamma * next_max_q\
                                                                      - self.action_value[tabular_state][index])
                obs = next_obs
                tabular_state = next_tarbular_state
                total_reward += reward

                if (done):
                    games_reward.append(total_reward)

            if (ep % 100) == 0:
                test_reward = self.run_episodes(self.env, 50)
                print("Learning Test Episode:{:5d}  Eps:{:2.4f}  Rew:{:2.4f}".format(ep, epsilon, test_reward))
                test_rewards.append(test_reward)
                episodes.append(ep)

        plt.plot(episodes, test_rewards, 'b-')  # 파란색 실선으로 그래프 그리기
        plt.xlabel('Episode')  # x축 레이블 설정
        plt.ylabel('Average Reward')  # y축 레이블 설정
        plt.title('Average Test Reward per 100 Episodes\nalpha={0}, initial_epsilon={1}, gamma = {2}, decay={3}'.format(alpha, initial_epsilon, gamma, decay))  # 그래프 제목 설정
        plt.show()  # 그래프 출력1

    def Test(self):
        env = pddlgym.make("PDDLEnvDroneTest-v0")
        test_reward = self.run_episodes(self.env, 10, True)
        print("Total Reward of 10 tests:{0}".format(test_reward))

    def __str__(self):
        str(self.state)
        
if __name__ == '__main__':

    env = pddlgym.make("PDDLEnvDrone-v0", seed = int(time.time()))
    rl = TabularRL(env)
    rl.Q_learning()
    rl.Test()