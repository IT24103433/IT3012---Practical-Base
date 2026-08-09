# agent.py
import random


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        if percept.get('food_here'):
            return 'Up'
        if percept.get('wall_ahead'):
            return 'Right'
        return random.choice(self.actions_pool)


class SimpleReflexAgent:
    """A reactive agent that only uses the current percept."""

    def sense_and_act(self, percept: dict) -> str:
        if percept.get('food_here'):
            return 'Up'
        if percept.get('wall_ahead'):
            return 'Left'
        return 'Right'


class ModelBasedAgent:
    """A reactive agent that uses a small internal memory to avoid loops."""

    def __init__(self):
        self.last_action = None
        self.last_percept = None
        self.visited_states = set()

    def sense_and_act(self, percept: dict) -> str:
        state = (
            bool(percept.get('wall_ahead')),
            bool(percept.get('food_here')),
            self.last_action,
        )

        if state in self.visited_states:
            if percept.get('wall_ahead'):
                action = 'Right' if self.last_action != 'Right' else 'Down'
            else:
                action = 'Left' if self.last_action != 'Left' else 'Up'
        else:
            if percept.get('food_here'):
                action = 'Up'
            elif percept.get('wall_ahead'):
                action = 'Left'
            else:
                action = 'Right'
            self.visited_states.add(state)

        self.last_percept = percept
        self.last_action = action
        return action


class SearchAgent:
    """A problem-solving agent that uses breadth-first search on a static grid."""

    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        start = (start_pos[0], start_pos[1])
        goal = (goal_pos[0], goal_pos[1])
        width, height = grid_size

        if start == goal:
            return []

        frontier = [start]
        parent = {start: None}
        action_map = {}
        directions = {
            'Up': (0, 1),
            'Down': (0, -1),
            'Left': (-1, 0),
            'Right': (1, 0),
        }

        while frontier:
            current = frontier.pop(0)
            if current == goal:
                break

            x, y = current
            for action, (dx, dy) in directions.items():
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if next_pos in walls or next_pos in parent:
                    continue
                parent[next_pos] = current
                action_map[next_pos] = action
                frontier.append(next_pos)

        if goal not in parent:
            return []

        path = []
        current = goal
        while parent[current] is not None:
            action = action_map[current]
            path.append(action)
            current = parent[current]
        path.reverse()
        return path
