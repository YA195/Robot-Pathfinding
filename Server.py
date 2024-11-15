from flask import Flask, render_template, jsonify, request
from collections import deque
import random
import time  

app = Flask(__name__, template_folder='templates', static_folder='static')

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

def bfs(start, goal):
    queue = deque([start])
    visited = {start}
    parent = {start: None}

    while queue:
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
    return path[::-1] if path and path[-1] == start else []

def dfs(start, goal):
    stack = [start]
    visited = {start}
    parent = {start: None}

    while stack:
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
    return path[::-1] if path and path[-1] == start else []

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
    if algorithm == 'bfs':
        path = bfs(start, goal_position)
    elif algorithm == 'dfs':
        path = dfs(start, goal_position)
    else:
        return jsonify({"error": "Algorithm not found"}), 404
    end_time = time.time()
    execution_time = int((end_time - start_time) * 1000)

    performance = {
        "time": execution_time,
        "space": len(path),
        "optimality": "Yes" if path else "No",
        "completeness": "Yes" if path else "No"
    }

    return jsonify({"path": path, "cost": len(path), "performance": performance})

@app.route('/generate-map', methods=['GET'])
def generate_map_route():
    global map_layout
    map_layout = generate_map()
    return jsonify({"map_layout": map_layout})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
