# agent.py
import random
import heapq
import math
from collections import deque


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
    """A problem-solving agent that plans paths on a static grid."""

    def __init__(self):
        self.plan = []
        self.active_algo = 'AStar'

    def manhattan_distance(self, pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def euclidean_distance(self, pos, goal):
        return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan'):
        start = (start_pos[0], start_pos[1])
        goal = (goal_pos[0], goal_pos[1])
        width, height = grid_size
        walls = set(walls)

        if heuristic_type == 'euclidean':
            heuristic = self.euclidean_distance
        else:
            heuristic = self.manhattan_distance

        frontier = [(heuristic(start, goal), 0, start, [])]
        reached_states = set()
        directions = {
            'Up': (0, 1),
            'Down': (0, -1),
            'Left': (-1, 0),
            'Right': (1, 0),
        }

        while frontier:
            f_cost, g_cost, current_pos, path_taken = heapq.heappop(frontier)
            if current_pos == goal:
                return path_taken
            if current_pos in reached_states:
                continue
            reached_states.add(current_pos)

            x, y = current_pos
            for action, (dx, dy) in directions.items():
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if next_pos in walls or next_pos in reached_states:
                    continue
                new_g_cost = g_cost + 1
                new_f_cost = new_g_cost + heuristic(next_pos, goal)
                heapq.heappush(
                    frontier,
                    (new_f_cost, new_g_cost, next_pos, path_taken + [action]),
                )

        return []

    def _search(self, start_pos, goal_pos, walls, grid_size, frontier_type):
        start = (start_pos[0], start_pos[1])
        goal = (goal_pos[0], goal_pos[1])
        width, height = grid_size

        if start == goal:
            return []

        directions = {
            'Up': (0, 1),
            'Down': (0, -1),
            'Left': (-1, 0),
            'Right': (1, 0),
        }
        reached = {start}
        parent = {start: None}
        action_map = {}
        counter = 0

        if frontier_type == 'bfs':
            frontier = deque([start])
            pop_node = frontier.popleft
            push_node = frontier.append
        elif frontier_type == 'dfs':
            frontier = [start]
            pop_node = frontier.pop
            push_node = frontier.append
        else:
            frontier = [(0, counter, start)]

            def pop_node():
                return heapq.heappop(frontier)[2]

            def push_node(node):
                nonlocal counter
                counter += 1
                heapq.heappush(frontier, (parent_cost[node], counter, node))

        parent_cost = {start: 0}
        while frontier:
            current = pop_node()
            if current == goal:
                break

            x, y = current
            for action, (dx, dy) in directions.items():
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if next_pos in walls or next_pos in reached:
                    continue
                reached.add(next_pos)
                parent[next_pos] = current
                action_map[next_pos] = action
                parent_cost[next_pos] = parent_cost[current] + 1
                push_node(next_pos)

        return self._reconstruct_path(goal, parent, action_map)

    @staticmethod
    def _reconstruct_path(goal, parent, action_map):
        if goal not in parent:
            return []

        path = []
        current = goal
        while parent[current] is not None:
            path.append(action_map[current])
            current = parent[current]
        path.reverse()
        return path

    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        return self._search(start_pos, goal_pos, set(walls), grid_size, 'bfs')

    def dfs_search(self, start_pos, goal_pos, walls, grid_size):
        return self._search(start_pos, goal_pos, set(walls), grid_size, 'dfs')

    def ucs_search(self, start_pos, goal_pos, walls, grid_size):
        return self._search(start_pos, goal_pos, set(walls), grid_size, 'ucs')

    def sense_and_act(self, percept: dict) -> str:
        if not self.plan:
            food_positions = [tuple(food)
                              for food in percept.get('all_food', [])]
            start_pos = tuple(percept.get('agent_pos', (0, 0)))
            food_positions = [
                food for food in food_positions if food != start_pos]

            if food_positions:
                goal_pos = min(
                    food_positions,
                    key=lambda food: abs(
                        food[0] - start_pos[0]) + abs(food[1] - start_pos[1]),
                )
                search_methods = {
                    'BFS': self.bfs_search,
                    'DFS': self.dfs_search,
                    'UCS': self.ucs_search,
                }
                if self.active_algo == 'AStar':
                    self.plan = self.astar_search(
                        start_pos,
                        goal_pos,
                        percept.get('walls', []),
                        percept.get('grid_size', (0, 0)),
                    )
                else:
                    search_method = search_methods.get(
                        self.active_algo, self.bfs_search)
                    self.plan = search_method(
                        start_pos,
                        goal_pos,
                        percept.get('walls', []),
                        percept.get('grid_size', (0, 0)),
                    )

        return self.plan.pop(0) if self.plan else 'Up'
