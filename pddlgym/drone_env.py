from core import PDDLEnv
from tabular_rl_advised import InvalidDirection
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import io

ICON_SIZE = 90

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
    

class GridObservation():
    def __init__(self, grid_size = 5, num_threat = 1):
        self.grid_size = grid_size
        self.num_threat = num_threat
        self.drone_x = 0
        self.drone_y = 0
        self.drone_to = ''
        self.goal_x = 0
        self.goal_y = 0
        self.threats = []
        self.rel_state = ()     # the Drone from the Goal, the threats from the drone
        self.drone_icon_north = Image.open('pddlgym/drone.png').resize((ICON_SIZE, ICON_SIZE))
        self.drone_icon_east = self.drone_icon_north.rotate(270)
        self.drone_icon_south = self.drone_icon_north.rotate(180)
        self.drone_icon_west = self.drone_icon_north.rotate(90)
        self.threat_icon = Image.open('pddlgym/threat.png').resize((ICON_SIZE, ICON_SIZE))
        self.goal_icon = Image.open('pddlgym/goal.png').resize((ICON_SIZE, ICON_SIZE))
    def reset(self):
        self.drone_x = 0
        self.drone_y = 0
        self.drone_to = ''
        self.goal_x = 0
        self.goal_y = 0
        self.threats = []
        self.rel_state = ()

    def render(self):
        # 그리드 크기
        grid_width = self.grid_size
        grid_height = self.grid_size

        # 셀 크기
        cell_width = 100
        cell_height = 100

        # 그리드 생성
        grid = [[None] * grid_width for _ in range(grid_height)]

        # 그래프 객체 생성
        fig, ax = plt.subplots(figsize=(10, 10))

        # 축 범위 설정
        ax.set_xlim(0, grid_width * cell_width)
        ax.set_ylim(0, grid_width * cell_height)
        ax.set_xticklabels([])  # x축 눈금 수치 없애기
        ax.set_yticklabels([])  # y축 눈금 수치 없애기
        # 그리드 그리기
        ax.grid(True, linestyle='-', linewidth=1, color='black')
        # 각 위치에 아이콘 배치
        for x in range(1, grid_width+1):
            for y in range(1, grid_height+1):
                # 셀 중심 좌표 계산
                cell_center_x = (x-1) * cell_width + cell_width / 2
                cell_center_y = (y-1) * cell_height + cell_height / 2

                # 드론 위치에 드론 아이콘 배치
                if (x, y) == (self.drone_x, self.drone_y):
                    if (self.drone_to == 'north'):
                        icon = self.drone_icon_north
                    elif(self.drone_to == 'east'):
                        icon = self.drone_icon_east
                    elif(self.drone_to == 'south'):
                        icon = self.drone_icon_south
                    else:
                        icon = self.drone_icon_west
                # 목표 위치에 목표 아이콘 배치
                elif (x, y) == (self.goal_x, self.goal_y):
                    icon = self.goal_icon
                # 위협 위치에 위협 아이콘 배치
                elif (x, y) in self.threats:
                    icon = self.threat_icon
                else:
                    continue

                # 아이콘을 그리드의 셀 중심에 배치
                left = cell_center_x - ICON_SIZE / 2
                top = cell_center_y - ICON_SIZE / 2
                ax.imshow(icon, extent=[left, left + ICON_SIZE, top, top + ICON_SIZE])


        # 그림을 이미지 객체로 변환
        canvas = FigureCanvas(fig)
        buffer = io.BytesIO()
        canvas.print_png(buffer)
        buffer.seek(0)
        image = Image.open(buffer)
        plt.close(fig)

        return image

    # PDDL Observation을 Grid의 Orient, Origin을 North, Drone 위치를 중심으로 좌표 변환하여 Tuple로 만들어 반환
    def convert_obs(self, obs):
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
        self.drone_x = int(drone_at._str.split(':')[0].split('-')[2])
        self.drone_y = int(drone_at._str.split(':')[0].split('-')[3])
        self.drone_to = drone_to._str.split('(')[1].split(':')[0]
        #get goal,threat position
        self.goal_x = int(obs[2]._str.split(':')[0].split('-')[2])
        self.goal_y = int(obs[2]._str.split(':')[0].split('-')[3])
        #transform coordinate by direction        
        transformed_drone = rotate_coordinate((self.drone_x, self.drone_y), self.drone_to, self.grid_size)
        transformed_goal = rotate_coordinate((self.goal_x, self.goal_y), self.drone_to, self.grid_size)
        transformed_threats = []
        threat_x = []
        threat_y = []
        for i in range(0, self.num_threat):
            threat_x.append(int(threat_at[i]._str.split(':')[0].split('-')[2]))
            threat_y.append(int(threat_at[i]._str.split(':')[0].split('-')[3]))
            self.threats.append((threat_x[i], threat_y[i]))
            transformed_threat = rotate_coordinate((threat_x[i], threat_y[i]), self.drone_to, self.grid_size)
            transformed_threats.append(transformed_threat)
        # if (self.num_target > 0):
        #     target_x = int(target_at._str.split(':')[0].split('-')[2])
        #     target_y = int(target_at._str.split(':')[0].split('-')[3])
        #     transformed_target = rotate_coordinate((target_x, target_y), drone_to, self.grid_size)
        
        #get relative coordinate from drone to goal
   
        #the agent from the goal
        self.rel_state = (transformed_drone[0] - transformed_goal[0],
                         transformed_drone[1] - transformed_goal[1])
 

        #an obstacle from the agent
        for i in range(0, self.num_threat):
            self.rel_state = (*self.rel_state, 
                             transformed_threats[i][0]-transformed_drone[0], 
                             transformed_threats[i][1]-transformed_drone[1] )
        # if (self.num_target > 0):
        #     if (not self.target_acquired):
        #         if (transformed_target[0] == transformed_drone[0]) and (transformed_target[1] == transformed_drone[1]):
        #             self.target_acquired = True
        #     converted_obs = (*converted_obs, *transformed_target, self.target_acquired)
        
        return self.rel_state
    
    #def render():
        
    
    
class DroneEnv(PDDLEnv):
    def __init__(self, domain_file, problem_dir, render=None, seed=0,
                 raise_error_on_invalid_action=False,
                 operators_as_actions=False,
                 dynamic_action_space=False,
                 grid_size=5, num_threat = 1, num_target = 0):
        super().__init__(domain_file, problem_dir, render, seed,
                 raise_error_on_invalid_action,
                 operators_as_actions,
                 dynamic_action_space)
        self.grid = GridObservation(grid_size, num_threat)
        self.num_threat = num_threat
        self.tabular_obs = ()

    def reset(self):
        self.grid.reset()
        self.obs, _ = super().reset()
        self.tabular_obs = self.grid.convert_obs(self.obs)
        return self.tabular_obs, None 
    
    
    def make_action(self, action):
        valid_actions = list(sorted(super().action_space.all_ground_literals(self.obs, valid_only=True)))

        if valid_actions is None :
            return None

        for valid_action in valid_actions:
            if valid_action.predicate.name == action:
                return valid_action
        
        #print('No action applied')
        return None


    def step(self, action):
        # action을 생성
        literal_action = self.make_action(action)
        
        if (literal_action is None):
            # invalid action면 이전 상태 그대로 반환
            return self.tabular_obs, -0.001, False, None
        else:
            # valid action이면 super().step(action) 후 grid 상태로 전화하여 리턴
            self.obs, reward, done, debug_info = super().step(literal_action)

            self.tabular_obs = self.grid.convert_obs(self.obs)

            return self.tabular_obs, reward, done, debug_info


    def print_obs(self):
        obs_str = 'Drone-At [{},{}][{}]  Goal-At [{}][{}] '.format(self.grid.drone_x, self.grid.drone_y, self.grid.drone_to, 
                                                               self.grid.goal_x, self.grid.goal_y)
        for i in range(0, self.num_threat):
            obs_str += 'Theat{}-At [{},{}] '.format(i+1, self.grid.threats[i][0], self.grid.threats[i][1])

        print(obs_str)

    def render(self):
        return self.grid.render()