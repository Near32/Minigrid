"""
This script allows you to manually control the simulator
using the keyboard arrows.
"""

import argparse
import math

import gymnasium as gym
import pyglet
from pyglet.window import key

import minigrid
#from minigrid.manual_control import ManualControl
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

# import sys

def str2bool(arg):
    if arg is None:
        return False
    if isinstance(arg, bool):
        return arg
    if isinstance(arg, str):
        if 'true' in arg.lower():
            return True
        else:
            return False
    else:
        raise NotImplementedError 


parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default='BabyAI-PickupDist-v0')
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--no-time-limit", action="store_true", help="ignore time step limits"
)
parser.add_argument("--num_objs", type=int, default=4)
parser.add_argument("--room_size", type=int, default=9)
parser.add_argument("--obs_size", type=int, default=7)

args = parser.parse_args()

env_kwargs = {}
if args.num_objs !=4:   env_kwargs['nbr_objs'] = args.num_objs
if args.room_size !=9:  env_kwargs['room_size'] = args.room_size
env_kwargs['maxNumRooms'] = 7

env = gym.make(
    args.env_name, 
    render_mode="human",
    #nbr_objs=args.num_objs,
    #room_size=args.room_size,
    agent_view_size=7,
    screen_size=640,
    tile_size=32,
    #maxNumRooms=7,
    **env_kwargs,
)

if args.no_time_limit:
    env.max_episode_steps = math.inf

env = RGBImgPartialObsWrapper(env, args.obs_size)
#env = ImgObsWrapper(env)


print("============")
print("Instructions")
print("============")
print("move: arrow keys\npickup: P\ndrop: D\ndone: ENTER\nquit: ESC")
print("============")

from minigrid.manual_control import ManualControl

manual_control = ManualControl(env, seed=args.seed)
manual_control.start()

'''

env.reset(seed=args.seed)

# Create the display window
env.render()


def step(action):
    global args 
    print(
        "step {}/{}: {}".format(
            env.step_count + 1, env.max_episode_steps, env.actions(action).name
        )
    )

    obs, reward, termination, truncation, info = env.step(action)

    if reward > 0:
        print(f"reward={reward:.2f}")

    if termination or truncation:
        print("done!")
        env.reset(seed=args.seed)
    
    env.render()

    if isinstance(obs, dict):
        if "mission" in obs:
            print(f"Mission: {obs['mission']}")

    

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        env.reset()
        env.render()
        return

    if symbol == key.ESCAPE:
        env.close()
        # sys.exit(0)

    if symbol == key.UP:
        step(env.actions.move_forward)
    elif symbol == key.DOWN:
        step(env.actions.move_back)

    elif symbol == key.LEFT:
        step(env.actions.turn_left)
    elif symbol == key.RIGHT:
        step(env.actions.turn_right)

    elif symbol == key.PAGEUP or symbol == key.P:
        step(env.actions.pickup)
    elif symbol == key.PAGEDOWN or symbol == key.D:
        step(env.actions.drop)

    elif symbol == key.ENTER:
        step(env.actions.done)


@env.unwrapped.window.event
def on_key_release(symbol, modifiers):
    pass


@env.unwrapped.window.event
def on_draw():
    env.render()


@env.unwrapped.window.event
def on_close():
    pyglet.app.exit()


# Enter main event loop
pyglet.app.run()

env.close()
'''

