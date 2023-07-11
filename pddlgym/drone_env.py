from pddlgym import PDDLEnv
from tabular_rl_advised import InvalidDirection

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
    
class DroneEnv(PDDLEnv):
    def __init__(self, domain_file, problem_dir, render=None, seed=0,
                 raise_error_on_invalid_action=False,
                 operators_as_actions=False,
                 dynamic_action_space=False,
                 grid_size=5, num_threat = 1, num_target = 0):
        PDDLEnv.__init__(self, domain_file, problem_dir, render, seed,
                 raise_error_on_invalid_action,
                 operators_as_actions,
                 dynamic_action_space)
        self.grid_size = grid_size
        self.num_threat = num_threat

    def step(self, action):
        # action을 valid action인지 확인
        # valid action이면 PDDLEnv.step(action) 후 grid 상태로 전화하여 리턴
        # invalid action이면 이전 상태 그대로 반환
        obs, reward, done, debug_info = PDDLEnv.step(action)
        tabular_obs = self._convert_obs(obs)
        return obs, reward, done, debug_info

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

        for i in range(0, self.num_threat):
            threat_x = int(threat_at[i]._str.split(':')[0].split('-')[2])
            threat_y = int(threat_at[i]._str.split(':')[0].split('-')[3])
            transformed_threat.append((threat_x, threat_y))
        # if (self.num_target > 0):
        #     target_x = int(target_at._str.split(':')[0].split('-')[2])
        #     target_y = int(target_at._str.split(':')[0].split('-')[3])
        #     transformed_target = rotate_coordinate((target_x, target_y), drone_to, self.grid_size)
        
        #get relative coordinate from drone to goal
   
        #the agent from the goal
        converted_obs = (transformed_drone[0] - transformed_goal[0],
                         transformed_drone[1] - transformed_goal[1])

        #an obstacle from the agent
        for i in range(0, self.num_threat):
            converted_obs = (*converted_obs, 
                             transformed_threat[i][0]-transformed_drone[0], 
                             transformed_threat[i][1]-transformed_drone[1] )
        # if (self.num_target > 0):
        #     if (not self.target_acquired):
        #         if (transformed_target[0] == transformed_drone[0]) and (transformed_target[1] == transformed_drone[1]):
        #             self.target_acquired = True
        #     converted_obs = (*converted_obs, *transformed_target, self.target_acquired)
        
        return converted_obs