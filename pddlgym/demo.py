"""Demonstrates basic PDDLGym usage with random action sampling
"""
import matplotlib; matplotlib.use('agg') # For rendering

from pddlgym.utils import run_demo
import pddlgym

def demo_random(env_name, render=True, problem_index=0, verbose=True):
    env = pddlgym.make("PDDLEnv{}-v0".format(env_name.capitalize()))
    env.fix_problem_index(problem_index)
    policy = lambda s : env.action_space.sample(s)
    video_path = "/tmp/{}_random_demo.mp4".format(env_name)
    run_demo(env, policy, render=render, verbose=verbose, seed=0,
             video_path=video_path)

def run_all(render=True, verbose=True):
    ## Some probabilistic environments
    demo_random("explodingblocks", render=render, verbose=verbose)
    demo_random("tireworld", render=render, verbose=verbose)
    demo_random("river", render=render, verbose=verbose)

    ## Some deterministic environments
    demo_random("sokoban", render=render, verbose=verbose)
    demo_random("gripper", render=render, verbose=verbose)
    demo_random("rearrangement", render=render, problem_index=6, verbose=verbose)
    demo_random("minecraft", render=render, verbose=verbose)
    demo_random("blocks", render=render, verbose=verbose)
    demo_random("blocks_operator_actions", render=render, verbose=verbose)
    # demo_random("quantifiedblocks", render=render, verbose=verbose)
    # demo_random("fridge", render=render, verbose=verbose)

def convert_tabular_obs(obs):
    for element in obs[0]:
        if (element.predicate.name == 'drone-at'):
            drone_at = element
        if (element.predicate.name == 'drone-to'):
            drone_to = element
    drone_x = int(drone_at._str.split(':')[0].split('-')[2])
    drone_y = int(drone_at._str.split(':')[0].split('-')[3])
    drone_to = drone_to._str.split('(')[1].split(':')[0]
    print(drone_to)
    goal_x = int(obs[2]._str.split(':')[0].split('-')[2])
    goal_y = int(obs[2]._str.split(':')[0].split('-')[3])
    print(goal_x)

if __name__ == '__main__':
    #run_all(render=False, verbose=True)
    env = pddlgym.make("PDDLEnvDrone-v0")
    env.fix_problem_index(0)
    policy = lambda s : env.action_space.sample(s)
    obs,_ = env.reset()

    convert_tabular_obs(obs)
    

    run_demo(env, policy, render=False, verbose=True, seed=0)