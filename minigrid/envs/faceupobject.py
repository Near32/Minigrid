from __future__ import annotations

from enum import IntEnum
import numpy as np
import math 
from functools import partial 

import gymnasium
from gymnasium.core import ActType, ObsType
from gymnasium import spaces
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.world_object import WorldObj, Lava, Goal, Door, Wall
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import ObjDesc, FaceUpInstr, ListInstr
from minigrid.minigrid_env import MissionSpace

# Object types we are allowed to describe in language
OBJ_TYPES = ["box", "ball", "key", "door"]
# Object types we are allowed to describe in language
OBJ_TYPES_NOT_DOOR = list(filter(lambda t: t != "door", OBJ_TYPES))


class BabyAIMissionSpace(MissionSpace):
    """
    Class that mimics the behavior required by minigrid.minigrid_env.MissionSpace,
    but does not change how missions are generated for BabyAI. It silences
    the gymnasium.utils.passive_env_checker given that it considers all strings to be
    plausible samples.
    """

    def __init__(self):
        super().__init__(mission_func=self._gen_mission)

    @staticmethod
    def _gen_mission():
        return "go"

    def contains(self, x: str):
        return True


class FaceUpActions(IntEnum):
    # rotate left, rotate right
    left = 0
    right = 1
    # Done completing task
    # done = 2 


class FaceUpObjDesc(ObjDesc):
    """
    Description of a set of objects in an environment, without 'the'/'a'...
    """
    
    def __init__(self, type, color=None):
        super().__init__(
            type=type,
            color=color,
            loc=None,
            relevant_types=OBJ_TYPES_NOT_DOOR,
            relevant_colors=COLOR_NAMES,
        )

    # Overriden from ObjDesc:
    def surface(self, env):
        """
        Generate a natural language representation of the object description
        """

        self.find_matching_objs(env)
        if len(self.obj_set)<=0:
            import ipdb; ipdb.set_trace()
            self.find_matching_objs(env)
        assert len(self.obj_set) > 0, "no object matching description"

        if self.type and self.color:
            s = self.color + " " + self.type
        elif self.type:
            s = str(self.type)
        elif self.color:
            s = self.color 

        '''
        # Singular vs plural
        if len(self.obj_set) > 1:
            s = "a " + s
        else:
            s = "the " + s
        '''
        return s

    
class FaceUpObjectEnv(RoomGridLevel):
    """

    ## Description

    Face up an object
    The object to face up is given by its type only, or
    by its color, or by its type and color.
    (in the current room, with distractors)
    All objects are placed in cells directly adjacent to the 
    middle cell of the room, where the agent is spawned.

    ## Mission Space

    "face up a/the {color}/{type}/{color}{type}"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {type} is the type of the object. Can be "ball", "box" or "key".

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Unused            |
    | 3   | pickup       | Unused            |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent faces up the object.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-FaceUpObject-v0`

    """

    def __init__(
        self, 
        nbr_objs:int=4,
        nbr_intermediate_states:int=0,
        nbr_wheels=1,
        debug:bool=False, 
        **kwargs,
    ):
        # Debug used to be for strict PickUp, i.e. failure when picking up wrong object.
        # TODO: it might be useful to simplify the current task...
        self.nbr_objs = nbr_objs
        assert self.nbr_objs < 12
        self.nbr_intermediate_states = nbr_intermediate_states
        assert self.nbr_intermediate_states < 5
        self.nbr_wheels = nbr_wheels
        self.debug = debug
        
        kwargs['room_size'] = self.nbr_wheels+self.nbr_objs+self.nbr_intermediate_states+3
        if kwargs['room_size']%2==0:    kwargs['room_size'] += 1
        kwargs['agent_view_size'] = self.nbr_wheels*2+1

        super().__init__(
            num_rows=1,
            num_cols=1,
            **kwargs,
        )        
        
        # Action enumeration for this environment
        self.actions = FaceUpActions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions)*self.nbr_wheels)
        
        achieved_goal_space = BabyAIMissionSpace()
        self.observation_space['achieved_goal'] = achieved_goal_space

        # Intermediates:
        self.id2inter_type = [Lava, Goal, partial(Door, color='grey')] 
        self.intermediate_count = {}
        for wheel_idx in range(self.nbr_wheels):
            self.intermediate_count[wheel_idx] = min(1, self.nbr_intermediate_states) 
        # i.e. after at most one move left, we find a goal state.
    
    def get_achieved_goal(self):
        wheel2obj = []
        for wheel_idx in range(self.nbr_wheels):
            if self.intermediate_count[wheel_idx] == 0:
                wheel2obj.append(
                    self.visible_objs[wheel_idx][0]
                )
            else:
                wheel2obj.append(None)
        achieved_goal = []
        for obj in wheel2obj:
            if obj is None:
                achieved_goal += ['none']*2
            else:
                achieved_goal += [obj.color, obj.type]

        achieved_goal = ' '.join(achieved_goal)
        return achieved_goal

    # Overriden from RoomGridLevel:
    def reset(self, **kwargs):
        return_info = kwargs.pop('return_info', True)
        obs, info = super().reset(**kwargs)

        # Recreate the verifier
        self.instrs.reset_verifier(self)

        # Compute the time step limit based on the maze size and instructions
        nav_time_room = self.room_size**2
        nav_time_maze = nav_time_room * self.num_rows * self.num_cols
        num_navs = self.num_navs_needed(self.instrs)

        if not self.fixed_max_steps:
            self.max_steps = num_navs * nav_time_maze
        
        obs['achieved_goal'] = self.get_achieved_goal()

        if return_info:
            return obs, info
        return obs

    def reset_grid(self, wheel_idx=0):
        for obj in self.visible_objs[wheel_idx]:
            if obj.cur_pos is None: continue
            self.grid.set(
                obj.cur_pos[0],
                obj.cur_pos[1],
                None,
            )
        for inter in self.intermediates[wheel_idx]:
            if inter.cur_pos is None:   continue
            self.grid.set(
                inter.cur_pos[0],
                inter.cur_pos[1],
                None,
            )

    def _put_intermediates(self, wheel_idx=0):
        nbr_intermediates = self.intermediate_count[wheel_idx]
        if nbr_intermediates < 0:   
            nbr_intermediates = 1+self.nbr_intermediate_states+self.intermediate_count[wheel_idx]
        midroom = self.room_size // 2
        for inter_idx, inter in enumerate(self.intermediates[wheel_idx]):
            if inter_idx < nbr_intermediates:
                #pose = [midroom + 1 + inter_idx, midroom]
                pose = [midroom + 1 + wheel_idx, midroom]
            else:
                # We need to put those intermediates outside of the view:
                pose = [1+wheel_idx,1+inter_idx]
            
            inter.cur_pos = pose
            self.put_obj(
                inter,
                pose[0],
                pose[1],
            )

    def _put_objs(self, wheel_idx=0):
        nbr_visible_objs = 4
        if self.nbr_intermediate_states != 0:   nbr_visible_objs = 2
        if self.intermediate_count[wheel_idx] != 0:    
            nbr_visible_objs = 0
        midroom = self.room_size // 2
        poses = [
            (midroom+1+wheel_idx, midroom), #right
            (midroom, midroom+1+wheel_idx), #bottom
            (midroom-1-wheel_idx, midroom), #left
            (midroom, midroom-1-wheel_idx), #top
        ]
        for obj_idx, obj in enumerate(self.visible_objs[wheel_idx]):
            if self.intermediate_count[wheel_idx] == 0:
                pose = poses[obj_idx]
            elif self.intermediate_count[wheel_idx] > 0:
                if obj_idx == 0:
                    pose = poses[-1]
                elif obj_idx == 1:
                    pose = poses[1]
                else:
                    # We need to put those objs outside of the view:
                    pose = [1+wheel_idx,self.room_size-1-obj_idx]
            elif self.intermediate_count[wheel_idx] < 0:
                if obj_idx == len(self.visible_objs[wheel_idx])-1:
                    pose = poses[-1]
                elif obj_idx == 0:
                    pose = poses[1]
                else:
                    # We need to put those objs outside of the view:
                    pose = [1+wheel_idx,self.room_size-1-obj_idx]
                    #pose = [1+obj_idx//midroom,1+obj_idx%midroom]

            obj.cur_pos = pose
            self.put_obj(
                obj,
                pose[0],
                pose[1],
            )
        
    def gen_mission(self):
        self.intermediate_count = {}
        for wheel_idx in range(self.nbr_wheels):
            self.intermediate_count[wheel_idx] = min(1, self.nbr_intermediate_states) 

        # The agent starts in the middle, facing right
        self.agent_pos = np.array(
            (
                (self.num_cols // 2) * (self.room_size - 1) + (self.room_size // 2),
                (self.num_rows // 2) * (self.room_size - 1) + (self.room_size // 2),
            )
        )
        self.agent_dir = 0

        # Add intermediates:
        self.intermediates = {}
        self.visible_objs = {}
        for wheel_idx in range(self.nbr_wheels):
            self.intermediates[wheel_idx] = [self.id2inter_type[inter_idx%len(self.id2inter_type)]() for inter_idx in range(self.nbr_intermediate_states)]
            # Add random objects in the room
            self.visible_objs[wheel_idx] = self.add_distractors(
                num_distractors=self.nbr_objs,
                all_unique=True,
            )
        
        for wheel_idx in range(self.nbr_wheels):
            self.reset_grid(wheel_idx=wheel_idx)
        for wheel_idx in range(self.nbr_wheels):
            self._put_intermediates(wheel_idx=wheel_idx)
            self._put_objs(wheel_idx=wheel_idx)

        self.instrs = []
        for wheel_idx in range(self.nbr_wheels):
            obj = self._rand_elem(self.visible_objs[wheel_idx])
            type = obj.type
            color = obj.color
            
            # For now, we always specify goals with both color and type:
            select_by = 'both' #self._rand_elem(["type", "color", "both"])
            if select_by == "color":
                type = None
            elif select_by == "type":
                color = None

            self.instrs.append(
                FaceUpInstr(FaceUpObjDesc(type, color), wheel_idx=wheel_idx)
            )
        
        self.instrs = ListInstr(self.instrs)
    
    # Overriding from RoomGridLevel/MiniGridEnv
    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        info = {}
        reward = 0
        terminated = False
        truncated = False

        need_rotate_objs_left = [False for _ in range(self.nbr_wheels)]
        need_rotate_objs_right = [False for _ in range(self.nbr_wheels)]

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)
        
        original_action = action
        action = original_action % 2
        wheel_idx = original_action // 2

        # Rotate left
        if action == self.actions.left:
            self.intermediate_count[wheel_idx] -= 1
        # Rotate right
        elif action == self.actions.right:
            self.intermediate_count[wheel_idx] += 1

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        
        # Regularisation:
        if self.intermediate_count[wheel_idx] > self.nbr_intermediate_states:
            self.intermediate_count[wheel_idx] = 0
            need_rotate_objs_right[wheel_idx] = True

        if self.intermediate_count[wheel_idx] < -self.nbr_intermediate_states:
            self.intermediate_count[wheel_idx] = 0
            need_rotate_objs_left[wheel_idx] = True

        # Update Representation:
        if need_rotate_objs_right[wheel_idx]:
            self.visible_objs[wheel_idx] = self.visible_objs[wheel_idx][1:]+[self.visible_objs[wheel_idx][0]]
        elif need_rotate_objs_left[wheel_idx]:
            self.visible_objs[wheel_idx] = [self.visible_objs[wheel_idx][-1]]+self.visible_objs[wheel_idx][:-1]
        
        # Generate Representation:
        self.reset_grid(wheel_idx=wheel_idx)
        self._put_objs(wheel_idx=wheel_idx)
        self._put_intermediates(wheel_idx=wheel_idx)

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        # If we've successfully completed the mission
        self.update_objs_poss()
        self.instrs.reset_verifier(self)
        status = self.instrs.verify(action)

        if status == "success":
            terminated = True
            reward = self._reward()
        elif status == "failure":
            terminated = True
            reward = 0

        obs['achieved_goal'] = self.get_achieved_goal()
        
        return obs, reward, terminated, truncated, info

    # Overriden from RoomGrid
    def add_distractors(
        self,
        i: int | None = None,
        j: int | None = None,
        num_distractors: int = 10,
        all_unique: bool = True,
    ) -> list[WorldObj]:
        """
        Add random objects that can potentially distract/confuse the agent.
        """
        # List of distractors added
        dists = []
        objs = []

        while len(dists) < num_distractors:
            color = self._rand_elem(COLOR_NAMES)
            type = self._rand_elem(["key", "ball", "box"])
            obj = (type, color)

            if all_unique and obj in objs:
                continue

            # Add the object to a random room if no room specified
            room_i = i
            room_j = j
            if room_i is None:
                room_i = self._rand_int(0, self.num_cols)
            if room_j is None:
                room_j = self._rand_int(0, self.num_rows)

            dist, pos = self.add_object(room_i, room_j, *obj)

            dists.append(dist)
            objs.append(obj)

        return dists    
