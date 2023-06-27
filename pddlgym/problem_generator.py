import random
from random import randint
import os

INITIAL_FIXED_POSITON = '\t\tdrone-at pos-3-1\n'
INITIAL_FIXED_DIRECTION = 't\t drone-to east\n'

DIR_LIST = ['north', 'east', 'south', 'west']
HEADING_LIST = ['right', 'left']

class ProblemGenerator():
    def __init__(self):
        self.domain_name = 'drone'
        self.grid_size = 5
        self.problem_size = 50
        self.is_fixed_goal = True
        self.is_fixed_init = False  # type: bool
        self.problem_number = 0

    def random_threat(self):
        while True:
            self.threat_x = randint(2, self.grid_size-1)
            self.threat_y = randint(2, self.grid_size-1)
            if (self.threat_x == self.goal_x) and (self.threat_y == self.goal_y):
                continue
            else:
                break
    
    def random_agent(self):
        while True:
            self.init_x = randint(1, self.grid_size)
            self.init_y = randint(1, self.grid_size)
            if (self.init_x == self.goal_x) and (self.init_y == self.goal_y):
                continue
            elif (self.init_x == self.threat_x) and (self.init_y == self.threat_y):
                continue
            else:
                break

    def generate_problem(self, index=0, learning = 200, test = 100):
        for index in range(0,learning):
            if not self.is_fixed_goal:
                self.goal_x = 3
                self.goal_y = 5
            else:
                self.goal_x = randint(1,5)
                self.goal_y = randint(1,5)
            self.random_threat()
            filename = './pddlgym/pddl/drone/problem{0}.pddl'.format(index)
            f = open(filename, 'w')
            f.write('(define (problem drone)(:domain drone)\n')
            f.write(self.generate_objects())
            f.write(self.generate_init())
            f.write(self.generate_goal())
            f.write(')')
            f.close()

        for index in range(0,test):
            if not self.is_fixed_goal:
                self.goal_x = 3
                self.goal_y = 5
            else:
                self.goal_x = randint(1,5)
                self.goal_y = randint(1,5)
            self.random_threat()
            filename = './pddlgym/pddl/drone_test/problem{0}.pddl'.format(index)
            f = open(filename, 'w')
            f.write('(define (problem drone)(:domain drone)\n')
            f.write(self.generate_objects())
            f.write(self.generate_init())
            f.write(self.generate_goal())
            f.write(')')
            f.close()
        

    def generate_objects(self):
        objects_string = '\t(:objects\n'
        for x in range(1, self.grid_size + 1):
            for y in range(1, self.grid_size + 1):
                objects_string += '\t\t'
                objects_string += 'pos-{0}-{1} - position\n'.format(x, y)
        objects_string += '\t)\n'
        return objects_string

    def generate_init(self):
        init_string = '\t(:init\n'
        if (self.is_fixed_init):
            init_string += INITIAL_FIXED_POSITON
            init_string += INITIAL_FIXED_DIRECTION
        else:
            self.random_agent()
            init_string += '\t\t(drone-at pos-{0}-{1})\n'.format(self.init_x, self.init_y)
            init_string += '\t\t(drone-to {0})\n'.format(DIR_LIST[randint(0, 3)])
            init_string += '\t\t(threat-at pos-{0}-{1})\n'.format(self.threat_x, self.threat_y)
                   
        init_string += self.generate_static_predicates()
        init_string += '\t)\n'
        return init_string

    def generate_static_predicates(self):
        static_predicates = ''
        for x in range(1, self.grid_size + 1):
            for y in range(1, self.grid_size + 1):
                for dir in DIR_LIST:
                    destination = self.move_forward(x, y, dir)
                    if (destination):
                        static_predicates += '\t\t(movable-forward pos-{0}-{1} '.format(x, y) + destination + dir + ')\n'
                    destination, newdir = self.move_right(x, y, dir)
                    if (destination):
                        static_predicates += '\t\t(movable-right pos-{0}-{1} '.format(x, y) + destination + dir +' ' + newdir + ')\n'
                    destination, newdir = self.move_left(x, y, dir)
                    if (destination):
                        static_predicates += '\t\t(movable-left pos-{0}-{1} '.format(x, y) + destination + dir + ' ' + newdir + ')\n'
        return static_predicates

    def move_forward(self, x, y, dir):
        dest_x = dest_y = 0
        
        if (dir == 'north'):
            dest_x = x
            dest_y = y+1
        elif (dir == 'south'):
            dest_x = x
            dest_y = y-1
        elif (dir == 'west'):
            dest_x = x-1
            dest_y = y
        else:
            dest_x = x+1
            dest_y = y

        if ((dest_x > self.grid_size or dest_x < 1) or (dest_y > self.grid_size or dest_y < 1)):
            return None
        else:
            return 'pos-{0}-{1}'.format(dest_x, dest_y) + ' '

    def move_right(self, x,y, dir) :
        dest_x = dest_y = 0
        if (dir == 'north'):
            dest_x = x+1
            dest_y = y
            new_dir = DIR_LIST[(DIR_LIST.index(dir)+1)%4]
        elif (dir == 'south'):
            dest_x = x-1
            dest_y = y
        elif (dir == 'west'):
            dest_x = x
            dest_y = y+1
        else:
            dest_x = x
            dest_y = y-1

        new_dir = DIR_LIST[(DIR_LIST.index(dir)+1)%4]
        if ((dest_x > self.grid_size or dest_x < 1) or (dest_y > self.grid_size or dest_y < 1)):
            return None, None
        else:
            return 'pos-{0}-{1}'.format(dest_x, dest_y) + ' ', new_dir

    def move_left(self, x,y, dir) :
        dest_x = dest_y = 0
        
        if (dir == 'north'):
            dest_x = x-1
            dest_y = y
        elif (dir == 'south'):
            dest_x = x+1
            dest_y = y
        elif (dir == 'west'):
            dest_x = x
            dest_y = y-1
        else:
            dest_x = x
            dest_y = y+1

        new_index = DIR_LIST.index(dir)-1
        if (new_index == -1):
            new_index = 3
        new_dir = DIR_LIST[new_index]

        if ((dest_x > self.grid_size or dest_x < 1) or (dest_y > self.grid_size or dest_y < 1)):
            return None, None
        else:
            return 'pos-{0}-{1}'.format(dest_x, dest_y) + ' ', new_dir
    
    def generate_goal(self):
        goal_string = ''
        if (self.is_fixed_goal):
            goal_string = '\t(:goal (drone-at pos-{}-{}))\n'.format(self.goal_x, self.goal_y)

        return goal_string

if __name__ == '__main__':
    pg = ProblemGenerator()
    pg.generate_problem()

