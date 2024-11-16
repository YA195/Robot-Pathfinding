from flask import Flask, render_template, jsonify, request
from collections import deque
import random
import time  
import heapq

app = Flask(__name__, template_folder='templates', static_folder='static')

# region Map

GRID_SIZE = (11, 11)
map_layout = [
    "sssssssssss",
    "swwwsswttws",
    "sttwsswttws",
    "sttwsswwwws",
    "swwwsssssss",
    "sssssssssss",
    "ssswwwsssss",
    "sssttwwwwws",
    "sssttwwttws",
    "ppswwwwttws",
    "ppsssssssss"
]

def generate_map():
    map_layout = [['s' for _ in range(11)] for _ in range(11)]
    
    for i in range(9, 11):
        for j in range(2):
            map_layout[i][j] = 'p'
    
    possible_positions = [
        (row, col)
        for row in range(2, 9)
        for col in range(2, 9)
    ]
    
    random.shuffle(possible_positions)
    
    target_tables = random.randint(7, 9)
    tables_placed = 0
    
    for row, col in possible_positions:
        if tables_placed >= target_tables:
            break

        buffer_zone_clear = all(
            map_layout[r][c] == 's'
            for r in range(row-1, row+2)
            for c in range(col-1, col+2)
            if 0 <= r < 11 and 0 <= c < 11
        )
        
        if not buffer_zone_clear:
            continue

        map_layout[row][col] = 't'
        open_sides = [
            ('top', row-1, col),
            ('bottom', row+1, col),
            ('left', row, col-1),
            ('right', row, col+1)
        ]
        
        possible_open_sides = [
            side for side, r, c in open_sides
            if 0 <= r < 11 and 0 <= c < 11 and map_layout[r][c] == 's'
        ]
        
        if len(possible_open_sides) >= 1:
            open_side = random.choice(possible_open_sides)
            
            for side, r, c in open_sides:
                if side != open_side:
                    map_layout[r][c] = 'w'
            
            tables_placed += 1

    return [''.join(row) for row in map_layout]

DIRECTIONS = [(0, 1), (0, -1), (-1, 0), (1, 0)]

def is_valid_move(y, x):
    if 0 <= x < GRID_SIZE[1] and 0 <= y < GRID_SIZE[0]:
        return map_layout[y][x] != 'w'
    return False

#endregion

# region Uninformed Search 
def bfs(start, goal):
    queue = deque([start])
    visited = {start}
    parent = {start: None}
    max_queue_size = 0  # Track memory usage

    while queue:
        max_queue_size = max(max_queue_size, len(queue))
        current = queue.popleft()
        if current == goal:
            break
        for dy, dx in DIRECTIONS:
            ny, nx = current[0] + dy, current[1] + dx
            if is_valid_move(ny, nx) and (ny, nx) not in visited:
                visited.add((ny, nx))
                parent[(ny, nx)] = current
                queue.append((ny, nx))

    path = []
    current = goal
    while current:
        path.append(current)
        current = parent.get(current)
    space = max_queue_size if path else 0
    return path[::-1] if path and path[-1] == start else [], space

def dfs(start, goal):
    stack = [start]
    visited = {start}
    parent = {start: None}
    max_stack_size = 0  # Track memory usage

    while stack:
        max_stack_size = max(max_stack_size, len(stack))
        current = stack.pop()
        if current == goal:
            break
        for dy, dx in DIRECTIONS:
            ny, nx = current[0] + dy, current[1] + dx
            if is_valid_move(ny, nx) and (ny, nx) not in visited:
                visited.add((ny, nx))
                parent[(ny, nx)] = current
                stack.append((ny, nx))

    path = []
    current = goal
    while current:
        path.append(current)
        current = parent.get(current)
    space = max_stack_size if path else 0
    return path[::-1] if path and path[-1] == start else [], space




def IDS(start, goal, limit):
    stack = [start]
    visited = set()
    visited.add(start)
    parent = {start: None}
    cost = {start: 0}

    while stack:
        current = stack.pop()
        if current == goal:
            break

        if cost[current] < limit:
            for direction in DIRECTIONS:
                next_y, next_x = current[0] + direction[0], current[1] + direction[1]
                if is_valid_move(next_y, next_x) and (next_y, next_x) not in visited:
                    visited.add((next_y, next_x))
                    parent[(next_y, next_x)] = current
                    cost[(next_y, next_x)] = cost[current] + 1
                    stack.append((next_y, next_x))

    path = []
    current = goal
    while current:
        path.append(current)
        current = parent.get(current)

    if path and path[-1] == start:
        return path[::-1], cost.get(goal, 0), len(visited)
    return [], 0, len(visited)

def iterative_deepening_search(start, goal):
    depth = 0
    while True:
        path, cost, space = IDS(start, goal, depth)
        if path:
            return path, cost, space
        depth += 1


def ucs(start, goal):
    frontier = [(0, start)]
    visited = set()
    parent = {start: None}
    cost = {start: 0}
    max_frontier_size = 0  # Track memory usage

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))  # Update max memory usage
        frontier.sort()
        current_cost, current = frontier.pop(0)

        if current in visited:
            continue

        visited.add(current)

        if current == goal:
            break

        for direction in DIRECTIONS:
            next_y, next_x = current[0] + direction[0], current[1] + direction[1]
            if is_valid_move(next_y, next_x):
                next_node = (next_y, next_x)
                new_cost = current_cost + 1

                if next_node not in cost or new_cost < cost[next_node]:
                    cost[next_node] = new_cost
                    parent[next_node] = current
                    frontier.append((new_cost, next_node))

    path = []
    current = goal
    while current:
        path.append(current)
        current = parent.get(current)

    return path[::-1] if path and path[-1] == start else [], max_frontier_size, cost.get(goal, 0)


# endregion

# region Heuristic Function

def heuristic_manhattan(current, goal):
    return abs(current[0] - goal[0]) + abs(current[1] - goal[1])


def heuristic_euclidean(current, goal):
    return ((current[0] - goal[0]) ** 2 + (current[1] - goal[1]) ** 2) ** 0.5


def greedy_best_first_search(start, goal, h_function):
    """Greedy Best-First Search."""
    frontier = [(h_function(start, goal), start)]
    visited = set()
    parent = {start: None}
    heuristic_values = {start: h_function(start, goal)}
    max_frontier_size = 0  # Track memory usage

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))  # Update max memory usage
        frontier.sort()
        heuristic_value, current = frontier.pop(0)

        if current == goal:
            path = []
            current_heuristics = []
            while current:
                path.append(current)
                current_heuristics.append(heuristic_values[current])
                current = parent[current]
            return path[::-1], current_heuristics[::-1], len(path) - 1, max_frontier_size

        if current in visited:
            continue

        visited.add(current)

        for direction in DIRECTIONS:
            next_y, next_x = current[0] + direction[0], current[1] + direction[1]
            if is_valid_move(next_y, next_x) and (next_y, next_x) not in visited:
                next_node = (next_y, next_x)

                h = h_function(next_node, goal)
                frontier.append((h, next_node))
                heuristic_values[next_node] = h

                if next_node not in parent:
                    parent[next_node] = current

    return [], [], 0, max_frontier_size


def a_star_search(start, goal, h_function):
    """A* Search."""
    frontier = [(0 + h_function(start, goal), start)]
    visited = set()
    parent = {start: None}
    g_values = {start: 0}
    f_values = {start: g_values[start] + h_function(start, goal)}
    h_values = {start: h_function(start, goal)}
    max_frontier_size = 0  # Track memory usage

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))  # Update max memory usage
        frontier.sort()
        f_value, current = frontier.pop(0)

        if current == goal:
            path = []
            current_heuristics = []
            current_costs = []
            current_f_values = []
            while current:
                path.append(current)
                current_heuristics.append(h_values[current])
                current_costs.append(g_values[current])
                current_f_values.append(f_values[current])
                current = parent[current]
            return (
                path[::-1],
                current_heuristics[::-1],
                current_costs[::-1],
                current_f_values[::-1],
                len(path) - 1,
                max_frontier_size,
            )

        visited.add(current)

        for direction in DIRECTIONS:
            next_y, next_x = current[0] + direction[0], current[1] + direction[1]
            if is_valid_move(next_y, next_x) and (next_y, next_x) not in visited:
                next_node = (next_y, next_x)

                g = g_values[current] + 1
                h = h_function(next_node, goal)
                f = g + h

                if next_node not in g_values or g < g_values[next_node]:
                    g_values[next_node] = g
                    f_values[next_node] = f
                    h_values[next_node] = h
                    parent[next_node] = current
                    frontier.append((f, next_node))

    return [], [], [], [], 0, max_frontier_size





# region Flask

@app.route('/')
def index():
    return render_template('index.html', map_layout=map_layout, path=[], goal=None, enumerate=enumerate)

goal_position = None

@app.route('/set_goal', methods=['POST'])
def set_goal():
    global goal_position
    goal_position = tuple(request.json.get('goal'))
    return jsonify({"message": "Goal set", "goal": goal_position})

@app.route('/path', methods=['GET'])
def get_path():
    algorithm = request.args.get('algorithm')
    start = (9, 0)
    global goal_position

    if goal_position is None:
        return jsonify({"error": "Goal not set"}), 400

    start_time = time.time()
    path = []
    space = 0
    heuristics = []
    current_cost = []
    f_value = []
    total_cost = 0

    try:
        if algorithm == 'bfs':
            path, space = bfs(start, goal_position)
            print("BFS space:", space)  # Debug statement
        elif algorithm == 'dfs':
            path, space = dfs(start, goal_position)
            print("DFS space:", space)  # Debug statement
        elif algorithm == 'ids':
            path, space, cost = iterative_deepening_search(start, goal_position)
            print("IDS space:", space)  # Debug statement
        elif algorithm == 'ucs':
            path, space, cost = ucs(start, goal_position)
            print("UCS space:", space)  # Debug statement
        elif algorithm == 'greedy_manhattan':
            path, heuristics, cost, space = greedy_best_first_search(start, goal_position, heuristic_manhattan)
            print("Greedy Manhattan space:", space)  # Debug statement
        elif algorithm == 'greedy_euclidean':
            path, heuristics, cost, space = greedy_best_first_search(start, goal_position, heuristic_euclidean)
            print("Greedy Euclidean space:", space)  # Debug statement
        elif algorithm == 'astar_euclidean':
            path, heuristics, current_cost, f_value, total_cost, space = a_star_search(start, goal_position, heuristic_euclidean)
            print("A* Euclidean space:", space)  # Debug statement
        elif algorithm == 'astar_manhattan':
            path, heuristics, current_cost, f_value, total_cost, space = a_star_search(start, goal_position, heuristic_manhattan)
            print("A* Manhattan space:", space)  # Debug statement

        else:
            return jsonify({"error": "Algorithm not found"}), 404
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    end_time = time.time()
    execution_time = int((end_time - start_time) * 1000)

    performance = {
        "time (ms)": execution_time,
        "space": space,
        "optimality": "Yes" if path else "No",
        "completeness": "Yes" if path else "No"
    }

    return jsonify({
        "path": path,
        "cost": len(path) if path else 0,
        "performance": performance
    })


@app.route('/generate-map', methods=['GET'])
def generate_map_route():
    global map_layout
    map_layout = generate_map()
    return jsonify({"map_layout": map_layout})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
 #endregion