from collections import defaultdict
from random import randint

import problem_generator



DIR_LIST = ['north', 'east', 'south', 'west']
ACT_LIST = ['heading-forward', 'heading-left', 'heading-right']

def transform_coordinate(coordinate, direction, grid_size = 5):
    if (direction=='east'):
        t_x = coordinate[1]
        t_y = grid_size - coordinate[0] + 1
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
    def __init__(self, env, default_value = -1000., grid_size=5):
        self.env = env
        self.state_value = defaultdict(lambda : default_value)
        self.action_value = defaultdict(lambda : default_value)
        self.action_space = env.action_space.predicates
        self.grid_size = grid_size
        self.obs , _ = env.reset()
        self.state = self._convert_obs(self.obs)
    
    def _convert_obs(self, obs):
        for element in obs[0]:
            if (element.predicate.name == 'drone-at'):
                drone_at = element
            if (element.predicate.name == 'drone-to'):
                drone_to = element
        
        self.drone_x = int(drone_at._str.split(':')[0].split('-')[2])
        self.drone_y = int(drone_at._str.split(':')[0].split('-')[3])
        self.drone_to = drone_to._str.split('(')[1].split(':')[0]
        #get goal position
        self.goal_x = int(obs[2]._str.split(':')[0].split('-')[2])
        self.goal_y = int(obs[2]._str.split(':')[0].split('-')[3])
        #transform coordinate by direction        
        transformed_drone = transform_coordinate((self.drone_x, self.drone_y), self.drone_to, self.grid_size)
        transformed_goal = transform_coordinate((self.goal_x, self.goal_y), self.drone_to, self.grid_size)
        #get relative coordinate from drone to goal
        relative_pos_to_goal =  (transformed_goal[0]-transformed_drone[0], transformed_goal[1]-transformed_drone[1])

        return (relative_pos_to_goal, DIR_LIST.index(self.drone_to))

    def sample_action(self):
        valid_actions = list(sorted(self.env.action_space.all_ground_literals(self.obs, valid_only=True)))
        random_action = valid_actions[randint(0, len(valid_actions) - 1)]
        return random_action

    def run_episode(self, policy = None):
        done = False
        if (not policy):
            while (True):
                print('Position({0},{1}) Direction {2}'.format(self.drone_x, self.drone_y, self.drone_to))
                action = self.sample_action()
                print(action)
                self.obs, reward, done, _ = self.env.step(action)
                self.state = self._convert_obs(self.obs)
                
                if (done):
                    print('----------Episode End-------------\n')
                    break
    def __str__(self):
        str(self.state)
        
