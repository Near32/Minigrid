"""
MiniGrid Environment for Duplicate Object Selection Tasks

This module implements environments where agents must identify and collect
one of two identical objects among a set of diverse objects.

Two primary variants are provided:
1. Color-focused: All objects share the same shape but differ in color,
   with exactly two objects sharing the target color.
2. Shape-focused: All objects share the same color but differ in shape,
   with exactly two objects sharing the target shape.
"""

from __future__ import annotations

from collections import Counter
import numpy as np
from gymnasium import spaces
from minigrid.core.constants import COLOR_NAMES, COLORS
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Key, WorldObj
from minigrid.minigrid_env import MiniGridEnv


class DuplicateObjectEnv(MiniGridEnv):
    """
    Base environment for duplicate object selection tasks.
    
    The agent is placed in a large empty room containing multiple objects.
    Exactly two objects share a common attribute (color or shape), and the
    agent must pick up either of these duplicate objects to succeed.
    
    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Parameters
    ----------
    room_size : int, optional
        Dimension of the square room (default: 16)
    num_objects : int, optional
        Total number of objects to place in the room (default: 4)
    duplicate_attribute : str, optional
        The attribute that defines duplicates: 'color' or 'shape' (default: 'color')
    max_steps : int, optional
        Maximum number of steps per episode (default: 1000)
    agent_view_size : int, optional
        Size of the agent's field of view (default: 3)
    """
    
    def __init__(
        self,
        room_size: int = 16,
        num_objects: int = 4,
        duplicate_attribute: str = 'color',
        max_steps: int | None = None,
        agent_view_size: int | None = None,
        **kwargs
    ):
        assert room_size >= 5, "Room size must be at least 5"
        assert num_objects >= 2, "Must have at least 2 objects"
        assert duplicate_attribute in ['color', 'shape'], \
            "duplicate_attribute must be 'color' or 'shape'"
        
        self.room_size = room_size
        self.num_objects = num_objects
        self.duplicate_attribute = duplicate_attribute
        
        # Define mission space
        mission_space = MissionSpace(mission_func=self._gen_mission) 

        if max_steps is None:
            max_steps = room_size ** 2
        
        if agent_view_size is None:
            agent_view_size = 3
        
        print(agent_view_size) 
        super().__init__(
            mission_space=mission_space,
            width=room_size,
            height=room_size,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            see_through_walls=False,
            **kwargs
        )
    
    @staticmethod
    def _gen_mission() -> str:
        return "pick up a duplicate object"
    
    def _gen_grid(self, width: int, height: int) -> None:
        """
        Generate the grid for the environment.
        
        This method is overridden in subclasses to implement specific
        object placement strategies.
        """
        raise NotImplementedError("Subclasses must implement _gen_grid")
    
    def _place_objects_randomly(
        self,
        objects: list[WorldObj]
    ) -> list[tuple[int, int]]:
        """
        Place objects at random positions in the grid.
        
        Parameters
        ----------
        objects : list[WorldObj]
            List of objects to place
        
        Returns
        -------
        list[tuple[int, int]]
            List of (x, y) positions where objects were placed
        """
        positions = []
        for obj in objects:
            pos = self.place_obj(obj)
            positions.append(pos)
        return positions
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Parameters
        ----------
        action : int
            Action to execute
        
        Returns
        -------
        tuple
            Observation, reward, terminated, truncated, info
        """
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Check if agent picked up a correct object
        if action == self.actions.pickup and self.carrying is not None:
            if self._is_correct_object(self.carrying):
                reward = self._reward()
                terminated = True
            else:
                # Picked up wrong object - terminate with no reward
                reward = 0
                terminated = True
        
        return obs, reward, terminated, truncated, info
    
    def _is_correct_object(self, obj: WorldObj) -> bool:
        """
        Check if the carried object is a correct duplicate.
        
        This method must be implemented by subclasses.
        
        Parameters
        ----------
        obj : WorldObj
            Object to check
        
        Returns
        -------
        bool
            True if object is a correct duplicate
        """
        raise NotImplementedError("Subclasses must implement _is_correct_object")


class ColorFocusedDuplicateEnv(DuplicateObjectEnv):
    """
    Color-focused duplicate object environment.
    
    All objects share the same shape but have different colors.
    Exactly two objects share the target color, and the agent must
    pick up one of them.
    
    Parameters
    ----------
    room_size : int, optional
        Dimension of the square room (default: 16)
    num_objects : int, optional
        Total number of objects to place (default: 4)
    object_shape : str, optional
        Shape of all objects: 'key', 'ball', or 'box' (default: 'key')
    duplicate_color : str, optional
        Color that appears twice; if None, randomly selected (default: None)
    available_colors : list[str], optional
        List of colors to use; if None, uses all MiniGrid colors (default: None)
    shape_palette : list[str], optional
        List of shapes to sample per object; defaults to [object_shape] (backwards-compatible)
    object_catalog : list[tuple[str, str]], optional
        Explicit list of (shape, color) pairs to instantiate instead of random sampling
    max_steps : int, optional
        Maximum number of steps per episode (default: room_size * 2)
    agent_view_size : int, optional
        Size of the agent's field of view (default: 3)
    """
    
    SHAPE_CLASSES = {
        'key': Key,
        'ball': Ball,
        'box': Box
    }
    
    def __init__(
        self,
        room_size: int = 16,
        num_objects: int = 4,
        object_shape: str = 'key',
        duplicate_color: str | None = None,
        available_colors: list[str] | None = None,
        shape_palette: list[str] | None = None,
        object_catalog: list[tuple[str, str]] | None = None,
        max_steps: int | None = None,
        agent_view_size: int | None = None,
        **kwargs
    ):
        assert object_shape in self.SHAPE_CLASSES, \
            f"object_shape must be one of {list(self.SHAPE_CLASSES.keys())}"
        
        self.object_shape = object_shape
        self.duplicate_color = duplicate_color
        self.available_colors = available_colors or list(COLOR_NAMES)
        self.shape_palette = list(shape_palette or [object_shape])
        assert self.shape_palette, "shape_palette must contain at least one shape"
        for shape in self.shape_palette:
            assert shape in self.SHAPE_CLASSES, \
                f"shape_palette entries must be one of {list(self.SHAPE_CLASSES.keys())}"
        
        self.object_catalog = object_catalog
        if self.object_catalog is not None:
            assert len(self.object_catalog) == num_objects, \
                "object_catalog must contain exactly num_objects entries"
            for shape, color in self.object_catalog:
                assert shape in self.SHAPE_CLASSES, \
                    f"object_catalog shape '{shape}' must be valid"
                assert color in COLOR_NAMES, \
                    f"object_catalog color '{color}' must be valid"
        
        # Validate duplicate_color if provided
        if duplicate_color is not None:
            assert duplicate_color in self.available_colors, \
                f"duplicate_color must be one of {self.available_colors}"
        
        # Ensure we have enough colors
        if self.object_catalog is None:
            assert len(self.available_colors) >= num_objects - 1, \
                f"Need at least {num_objects - 1} colors for {num_objects} objects"
        
        super().__init__(
            room_size=room_size,
            num_objects=num_objects,
            duplicate_attribute='color',
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            **kwargs
        )
        
        # Store the actual duplicate color used in the current episode
        self._current_duplicate_color = None
    
    def _gen_grid(self, width: int, height: int) -> None:
        """
        Generate grid with objects of the same shape but different colors.
        """
        # Create empty grid
        self.grid = Grid(width, height)
        
        # Generate outer walls
        self.grid.wall_rect(0, 0, width, height)

        using_catalog = self.object_catalog is not None

        # Select duplicate color for this episode
        if self.duplicate_color is not None:
            self._current_duplicate_color = self.duplicate_color
        elif not using_catalog:
            self._current_duplicate_color = self._rand_elem(self.available_colors)

        if using_catalog:
            obj_specs = list(self.object_catalog)
            if self._current_duplicate_color is None:
                color_counts = Counter(color for _, color in obj_specs)
                duplicate_colors = [
                    color for color, count in color_counts.items() if count >= 2
                ]
                assert duplicate_colors, \
                    "object_catalog must contain a color that appears at least twice"
                self._current_duplicate_color = self._rand_elem(duplicate_colors)
            else:
                duplicate_matches = sum(
                    1 for _, color in obj_specs if color == self._current_duplicate_color
                )
                assert duplicate_matches >= 2, \
                    "object_catalog must contain at least two objects with duplicate_color"
        else:
            # Select other colors (ensuring they're different from duplicate)
            other_colors = [
                c for c in self.available_colors if c != self._current_duplicate_color
            ]
            selected_other_colors = self._rand_subset(
                other_colors,
                self.num_objects - 2
            )

            # Build list of all colors: duplicate appears twice, others once
            all_colors = [self._current_duplicate_color] * 2 + selected_other_colors
            self.np_random.shuffle(all_colors)
            sampled_shapes = [
                self._rand_elem(self.shape_palette) for _ in range(self.num_objects)
            ]
            obj_specs = list(zip(sampled_shapes, all_colors))

        # Create objects
        objects = [
            self.SHAPE_CLASSES[shape](color)
            for shape, color in obj_specs
        ]
        
        # Place objects randomly
        self._place_objects_randomly(objects)
        
        # Place agent at random position
        self.place_agent()
        
        # Set mission
        self.mission = self._gen_mission()
    
    def _is_correct_object(self, obj: WorldObj) -> bool:
        """
        Check if the object has the correct duplicate color.
        
        Parameters
        ----------
        obj : WorldObj
            Object to check
        
        Returns
        -------
        bool
            True if object has the duplicate color
        """
        return obj.color == self._current_duplicate_color


class ShapeFocusedDuplicateEnv(DuplicateObjectEnv):
    """
    Shape-focused duplicate object environment.
    
    All objects share the same color but have different shapes.
    Exactly two objects share the target shape, and the agent must
    pick up one of them.
    
    Parameters
    ----------
    room_size : int, optional
        Dimension of the square room (default: 16)
    num_objects : int, optional
        Total number of objects to place (default: 4)
    object_color : str, optional
        Color of all objects (default: 'red')
    duplicate_shape : str, optional
        Shape that appears twice; if None, randomly selected (default: None)
    available_shapes : list[str], optional
        List of shapes to use (default: ['key', 'ball', 'box'])
    color_palette : list[str], optional
        List of colors to sample per object; defaults to [object_color]
    object_catalog : list[tuple[str, str]], optional
        Explicit list of (shape, color) pairs to instantiate instead of sampling
    max_steps : int, optional
        Maximum number of steps per episode (default: room_size * 2)
    agent_view_size : int, optional
        Size of the agent's field of view (default: 3)
    """
    
    SHAPE_CLASSES = {
        'key': Key,
        'ball': Ball,
        'box': Box
    }
    
    def __init__(
        self,
        room_size: int = 16,
        num_objects: int = 4,
        object_color: str = 'red',
        duplicate_shape: str | None = None,
        available_shapes: list[str] | None = None,
        color_palette: list[str] | None = None,
        object_catalog: list[tuple[str, str]] | None = None,
        max_steps: int | None = None,
        agent_view_size: int | None = None,
        **kwargs
    ):
        assert object_color in COLOR_NAMES, \
            f"object_color must be one of {COLOR_NAMES}"
        
        self.object_color = object_color
        self.duplicate_shape = duplicate_shape
        self.available_shapes = available_shapes or list(self.SHAPE_CLASSES.keys())
        self.color_palette = list(color_palette or [object_color])
        assert self.color_palette, "color_palette must contain at least one color"
        for color in self.color_palette:
            assert color in COLOR_NAMES, \
                f"color_palette entries must be one of {COLOR_NAMES}"
        
        self.object_catalog = object_catalog
        if self.object_catalog is not None:
            assert len(self.object_catalog) == num_objects, \
                "object_catalog must contain exactly num_objects entries"
            for shape, color in self.object_catalog:
                assert shape in self.SHAPE_CLASSES, \
                    f"object_catalog shape '{shape}' must be valid"
                assert color in COLOR_NAMES, \
                    f"object_catalog color '{color}' must be valid"
        
        # Validate duplicate_shape if provided
        if duplicate_shape is not None:
            assert duplicate_shape in self.available_shapes, \
                f"duplicate_shape must be one of {self.available_shapes}"
        
        # Ensure we have enough shapes
        if self.object_catalog is None:
            assert len(self.available_shapes) >= num_objects - 1, \
                f"Need at least {num_objects - 1} shapes for {num_objects} objects"
        
        super().__init__(
            room_size=room_size,
            num_objects=num_objects,
            duplicate_attribute='shape',
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            **kwargs
        )
        
        # Store the actual duplicate shape used in the current episode
        self._current_duplicate_shape = None
    
    def _gen_grid(self, width: int, height: int) -> None:
        """
        Generate grid with objects of the same color but different shapes.
        """
        # Create empty grid
        self.grid = Grid(width, height)
        
        # Generate outer walls
        self.grid.wall_rect(0, 0, width, height)

        using_catalog = self.object_catalog is not None

        # Select duplicate shape for this episode
        if self.duplicate_shape is not None:
            self._current_duplicate_shape = self.duplicate_shape
        elif not using_catalog:
            self._current_duplicate_shape = self._rand_elem(self.available_shapes)

        if using_catalog:
            obj_specs = list(self.object_catalog)
            if self._current_duplicate_shape is None:
                shape_counts = Counter(shape for shape, _ in obj_specs)
                duplicate_shapes = [
                    shape for shape, count in shape_counts.items() if count >= 2
                ]
                assert duplicate_shapes, \
                    "object_catalog must contain a shape that appears at least twice"
                self._current_duplicate_shape = self._rand_elem(duplicate_shapes)
            else:
                duplicate_matches = sum(
                    1 for shape, _ in obj_specs if shape == self._current_duplicate_shape
                )
                assert duplicate_matches >= 2, \
                    "object_catalog must contain at least two objects with duplicate_shape"
        else:
            # Select other shapes (ensuring they're different from duplicate)
            other_shapes = [
                s for s in self.available_shapes if s != self._current_duplicate_shape
            ]
            selected_other_shapes = self._rand_subset(
                other_shapes,
                self.num_objects - 2
            )

            # Build list of all shapes: duplicate appears twice, others once
            all_shapes = [self._current_duplicate_shape] * 2 + selected_other_shapes
            self.np_random.shuffle(all_shapes)
            colors = [
                self._rand_elem(self.color_palette) for _ in range(self.num_objects)
            ]
            obj_specs = list(zip(all_shapes, colors))
        
        # Create objects
        objects = [
            self.SHAPE_CLASSES[shape](color)
            for shape, color in obj_specs
        ]
        
        # Place objects randomly
        self._place_objects_randomly(objects)
        
        # Place agent at random position
        self.place_agent()
        
        # Set mission
        self.mission = self._gen_mission()
    
    def _is_correct_object(self, obj: WorldObj) -> bool:
        """
        Check if the object has the correct duplicate shape.
        
        Parameters
        ----------
        obj : WorldObj
            Object to check
        
        Returns
        -------
        bool
            True if object has the duplicate shape
        """
        return obj.type == self._current_duplicate_shape


# Convenience factory functions

def create_color_focused_env(
    room_size: int = 16,
    num_objects: int = 4,
    object_shape: str = 'key',
    duplicate_color: str | None = None,
    **kwargs
) -> ColorFocusedDuplicateEnv:
    """
    Factory function to create a color-focused duplicate environment.
    
    Parameters
    ----------
    room_size : int, optional
        Dimension of the square room (default: 16)
    num_objects : int, optional
        Total number of objects (default: 4)
    object_shape : str, optional
        Shape of all objects (default: 'key')
    duplicate_color : str, optional
        Color that appears twice (default: None for random)
    **kwargs
        Additional arguments passed to ColorFocusedDuplicateEnv
    
    Returns
    -------
    ColorFocusedDuplicateEnv
        Configured environment instance
    
    Examples
    --------
    >>> env = create_color_focused_env(room_size=20, num_objects=6)
    >>> obs, info = env.reset()
    """
    return ColorFocusedDuplicateEnv(
        room_size=room_size,
        num_objects=num_objects,
        object_shape=object_shape,
        duplicate_color=duplicate_color,
        **kwargs
    )


def create_shape_focused_env(
    room_size: int = 16,
    num_objects: int = 4,
    object_color: str = 'red',
    duplicate_shape: str | None = None,
    **kwargs
) -> ShapeFocusedDuplicateEnv:
    """
    Factory function to create a shape-focused duplicate environment.
    
    Parameters
    ----------
    room_size : int, optional
        Dimension of the square room (default: 16)
    num_objects : int, optional
        Total number of objects (default: 4)
    object_color : str, optional
        Color of all objects (default: 'red')
    duplicate_shape : str, optional
        Shape that appears twice (default: None for random)
    **kwargs
        Additional arguments passed to ShapeFocusedDuplicateEnv
    
    Returns
    -------
    ShapeFocusedDuplicateEnv
        Configured environment instance
    
    Examples
    --------
    >>> env = create_shape_focused_env(room_size=20, num_objects=6)
    >>> obs, info = env.reset()
    """
    return ShapeFocusedDuplicateEnv(
        room_size=room_size,
        num_objects=num_objects,
        object_color=object_color,
        duplicate_shape=duplicate_shape,
        **kwargs
    )


if __name__ == "__main__":
    """
    Demonstration of both environment variants.
    """
    print("=" * 70)
    print("Color-Focused Duplicate Environment Demo")
    print("=" * 70)
    
    # Create color-focused environment
    color_env = create_color_focused_env(
        room_size=16,
        num_objects=4,
        object_shape='key',
        duplicate_color='red'  # Fixed duplicate color
    )
    
    obs, info = color_env.reset()
    print(f"Mission: {color_env.mission}")
    print(f"Room size: {color_env.room_size}x{color_env.room_size}")
    print(f"Number of objects: {color_env.num_objects}")
    print(f"Object shape: {color_env.object_shape}")
    print(f"Duplicate color: {color_env._current_duplicate_color}")
    print()
    
    # Random color selection
    color_env_random = create_color_focused_env(
        room_size=20,
        num_objects=6,
        object_shape='ball',
        duplicate_color=None  # Random duplicate color
    )
    
    obs, info = color_env_random.reset()
    print(f"Random color variant - Mission: {color_env_random.mission}")
    print(f"Duplicate color (randomly selected): {color_env_random._current_duplicate_color}")
    print()
    
    print("=" * 70)
    print("Shape-Focused Duplicate Environment Demo")
    print("=" * 70)
    
    # Create shape-focused environment
    shape_env = create_shape_focused_env(
        room_size=16,
        num_objects=4,
        object_color='red',
        duplicate_shape='key'  # Fixed duplicate shape
    )
    
    obs, info = shape_env.reset()
    print(f"Mission: {shape_env.mission}")
    print(f"Room size: {shape_env.room_size}x{shape_env.room_size}")
    print(f"Number of objects: {shape_env.num_objects}")
    print(f"Object color: {shape_env.object_color}")
    print(f"Duplicate shape: {shape_env._current_duplicate_shape}")
    print()
    
    # Random shape selection
    shape_env_random = create_shape_focused_env(
        room_size=20,
        num_objects=6,
        object_color='blue',
        duplicate_shape=None  # Random duplicate shape
    )
    
    obs, info = shape_env_random.reset()
    print(f"Random shape variant - Mission: {shape_env_random.mission}")
    print(f"Duplicate shape (randomly selected): {shape_env_random._current_duplicate_shape}")
    print()
    
    print("=" * 70)
    print("Environment creation successful!")
    print("=" * 70)
