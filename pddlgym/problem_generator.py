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

    def generate_problem(self, index=0):
        filename = './pddlgym/pddl/drone/problem{0}.pddl'.format(index)
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
            init_string += '\t\t(drone-at pos-{0}-{1})\n'.format(randint(1, 5), randint(1, 5))
            init_string += '\t\t( drone-to {0})\n'.format(DIR_LIST[randint(0, 3)])
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
            goal_string = '\t(:goal (drone-at pos-3-5))\n'
        
        return goal_string


pg = ProblemGenerator()
pg.generate_problem()

