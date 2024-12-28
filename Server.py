from flask import Flask, render_template, jsonify, request
from collections import deque, defaultdict
from flask_socketio import SocketIO

import random
import time  
import heapq
import math


app = Flask(__name__, template_folder='templates', static_folder='static')
socketio = SocketIO(app, cors_allowed_origins="*")

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
    max_queue_size = 0   

    while queue:
        max_queue_size = max(max_queue_size, len(queue))
        current = queue.popleft()
        if current in goal:
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
    max_stack_size = 0   

    while stack:
        max_stack_size = max(max_stack_size, len(stack))
        current = stack.pop()
        if current in goal:
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
        if current in goal:
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
    max_frontier_size = 0   

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))   
        frontier.sort()
        current_cost, current = frontier.pop(0)

        if current in visited:
            continue

        visited.add(current)

        if current in goal:
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
    max_frontier_size = 0   

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))   
        frontier.sort()
        heuristic_value, current = frontier.pop(0)

        if current in goal:
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
    max_frontier_size = 0   

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))   
        frontier.sort()
        f_value, current = frontier.pop(0)

        if current in goal:
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

# endregion






# region Local search 

def hill_climbing(start, goal, h_function):
    current = start
    visited = set()
    path = [current]
    max_space = 0   

    while current != goal:
        neighbors = []
        for direction in DIRECTIONS:
            next_y, next_x = current[0] + direction[0], current[1] + direction[1]
            if is_valid_move(next_y, next_x) and (next_y, next_x) not in visited:
                neighbors.append((next_y, next_x))

        max_space = max(max_space, len(visited) + len(neighbors) + len(path))   

        if not neighbors:
            return [], max_space

        best_neighbor = None
        best_h_value = float('inf')
        for neighbor in neighbors:
            h_value = h_function(neighbor, goal)
            if h_value < best_h_value:
                best_h_value = h_value
                best_neighbor = neighbor

        if best_h_value >= h_function(current, goal):
            return [], max_space

        visited.add(current)
        current = best_neighbor
        path.append(current)

    return path, max_space
    current = start
    visited = set()
    path = [current]

    while current != goal:
        neighbors = []
        for direction in DIRECTIONS:
            next_y, next_x = current[0] + direction[0], current[1] + direction[1]
            if is_valid_move(next_y, next_x) and (next_y, next_x) not in visited:
                neighbors.append((next_y, next_x))

        if not neighbors:
            return [], 0

        best_neighbor = None
        best_h_value = float('inf')
        for neighbor in neighbors:
            h_value = h_function(neighbor, goal)
            if h_value < best_h_value:
                best_h_value = h_value
                best_neighbor = neighbor

        if best_h_value >= h_function(current, goal):
            return [], 0

        visited.add(current)
        current = best_neighbor
        path.append(current)

    return path, len(path) - 1

def simulated_annealing(start, goal, h_function, initial_temp=1000, cooling_rate=0.995, max_iterations=1000):
    current = start
    best = start
    visited = set()
    path = [current]
    temperature = initial_temp
    max_space = 0   

    def acceptance_probability(current_h, neighbor_h, temperature):
        """Calculate the acceptance probability of a worse solution."""
        if neighbor_h < current_h:
            return 1.0
        return math.exp((current_h - neighbor_h) / temperature)

    for iteration in range(max_iterations):
        neighbors = []
        for direction in DIRECTIONS:
            next_y, next_x = current[0] + direction[0], current[1] + direction[1]
            if is_valid_move(next_y, next_x) and (next_y, next_x) not in visited:
                neighbors.append((next_y, next_x))

        max_space = max(max_space, len(visited) + len(neighbors) + len(path))   

        if not neighbors:
            return [], max_space

        next_node = random.choice(neighbors)
        current_h = h_function(current, goal)
        next_h = h_function(next_node, goal)

        if acceptance_probability(current_h, next_h, temperature) > random.random():
            current = next_node
            path.append(current)

            if next_h < h_function(best, goal):
                best = current

        temperature *= cooling_rate

        if current in goal:
            break

    return path, max_space

def create_initial_population(start, goal, population_size=100):
    population = []
    for _ in range(population_size):
        path = generate_random_path(start, goal)
        population.append(path)
    return population

def generate_random_path(start, goal):
    path = [start]
    current = start
    while current != goal:
        neighbors = []
        for direction in DIRECTIONS:
            next_y, next_x = current[0] + direction[0], current[1] + direction[1]
            if is_valid_move(next_y, next_x) and (next_y, next_x) not in path:
                neighbors.append((next_y, next_x))
        if neighbors:
            current = random.choice(neighbors)
            path.append(current)
        else:
            break
    return path

def fitness(path, goal):
    if not path:
        return 0

    last_position = path[-1]
    return 1 / (abs(last_position[0] - goal[0]) + abs(last_position[1] - goal[1]) + 1)

def select_parents(population, goal):
    selected_parents = random.sample(population, 2)
    selected_parents.sort(key=lambda x: fitness(x, goal), reverse=True)
    return selected_parents[0], selected_parents[1]

def crossover(parent1, parent2):
    crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(path, goal, mutation_rate=0.1):
    if random.random() < mutation_rate:
        mutation_index = random.randint(1, len(path) - 1)
        current = path[mutation_index]
        neighbors = []
        for direction in DIRECTIONS:
            next_y, next_x = current[0] + direction[0], current[1] + direction[1]
            if is_valid_move(next_y, next_x) and (next_y, next_x) not in path:
                neighbors.append((next_y, next_x))
        if neighbors:
            path[mutation_index] = random.choice(neighbors)
    return path

def evolve_population(population, goal, population_size=100, mutation_rate=0.1):
    new_population = []
    while len(new_population) < population_size:
        parent1, parent2 = select_parents(population, goal)
        child1, child2 = crossover(parent1, parent2)
        new_population.append(mutate(child1, goal, mutation_rate))
        new_population.append(mutate(child2, goal, mutation_rate))
    return new_population

def genetic_algorithm(start, goal, population_size=100, generations=1000, mutation_rate=0.1):
    population = create_initial_population(start, goal, population_size)
    best_solution = None
    best_fitness = 0
    max_space = len(population)   

    for generation in range(generations):
        population.sort(key=lambda x: fitness(x, goal), reverse=True)
        fittest_individual = population[0]
        fittest_fitness = fitness(fittest_individual, goal)

        if fittest_fitness > best_fitness:
            best_solution = fittest_individual
            best_fitness = fittest_fitness

        new_population = evolve_population(population, goal, population_size, mutation_rate)
        max_space = max(max_space, len(new_population))   

        population = new_population

        if best_solution and best_solution[-1] == goal:
            break

    return best_solution, max_space
    population = create_initial_population(start, goal, population_size)
    best_solution = None
    best_fitness = 0

    for generation in range(generations):
        population.sort(key=lambda x: fitness(x, goal), reverse=True)
        fittest_individual = population[0]
        fittest_fitness = fitness(fittest_individual, goal)

        if fittest_fitness > best_fitness:
            best_solution = fittest_individual
            best_fitness = fittest_fitness

        population = evolve_population(population, goal, population_size, mutation_rate)

        if best_solution[-1] == goal:
            break

    return best_solution, best_fitness

# endregion
 


def get_reward(current_state, next_state, goal, path_length):
    current_distance = abs(current_state[0] - goal[0]) + abs(current_state[1] - goal[1])
    next_distance = abs(next_state[0] - goal[0]) + abs(next_state[1] - goal[1])
    
    # Base rewards
    if next_state == goal:
        return 100  # Reduced from 2000 to balance values
    
    if not is_valid_move(next_state[0], next_state[1]):
        return -100  # Reduced from -1000
    
    # Movement rewards
    distance_improvement = current_distance - next_distance
    return 20 * distance_improvement - 1  # Small penalty for each step
def q_learning(start, goals, episodes=1000, alpha=0.1, gamma=0.9, epsilon_start=1.0):
    Q = defaultdict(lambda: defaultdict(float))
    actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    best_path = []
    max_space = 0
    total_reward = 0
    current_pos = start
    final_path = []
    training_data = []
    epsilon = epsilon_start
    midway_point = episodes // 2
    q_value_snapshots = []

    for goal in goals:
        for episode in range(episodes):
            current_state = current_pos
            path = [current_state]
            episode_reward = 0
            visited = {current_state}
            
            if episode == midway_point or episode == episodes - 1:
                serializable_q_values = {str(k): {str(a): v for a, v in val.items()} 
                                      for k, val in Q.items()}
                policy = {str(k): str(v) for k, v in analyze_current_policy(Q).items()}
                q_value_snapshots.append({
                    'episode': episode,
                    'q_values': serializable_q_values,
                    'policy': policy
                })
            
            while current_state != goal and len(path) < 100:
                valid_actions = [action for action in actions 
                               if is_valid_move(current_state[0] + action[0], 
                                              current_state[1] + action[1])]
                
                if not valid_actions:
                    break
                
                action = random.choice(valid_actions) if random.random() < epsilon \
                        else max(valid_actions, key=lambda a: Q[current_state][a])
                
                next_state = (current_state[0] + action[0], current_state[1] + action[1])
                reward = get_reward(current_state, next_state, goal, len(path))
                
                next_max = max([Q[next_state][a] for a in valid_actions], default=0)
                Q[current_state][action] += alpha * (reward + gamma * next_max - Q[current_state][action])
                
                current_state = next_state
                path.append(current_state)
                visited.add(current_state)
                episode_reward += reward
                max_space = max(max_space, len(visited))

            training_data.append({
                'episode': episode,
                'reward': float(episode_reward),
                'exploration_rate': float(epsilon),
                'states_visited': len(visited),
                'path_length': len(path),
                'avg_q_value': float(calculate_avg_q_value(Q)),
                'policy_stability': float(calculate_policy_stability(Q))
            })
            
            epsilon *= 0.995

            if current_state == goal and (not best_path or len(path) < len(best_path)):
                best_path = path.copy()
                total_reward += episode_reward

        if best_path:
            final_path.extend(best_path if goal == goals[-1] else best_path[:-1])
            current_pos = goal
            best_path = []

    q_table_data = {str(state): {
        'up': float(Q[state][(-1, 0)]),
        'down': float(Q[state][(1, 0)]),
        'left': float(Q[state][(0, -1)]),
        'right': float(Q[state][(0, 1)])
    } for state in Q}

    return final_path, max_space, float(total_reward), q_table_data, training_data, q_value_snapshots

def analyze_current_policy(Q):
    policy = {}
    for state in Q:
        policy[state] = max(Q[state].items(), key=lambda x: x[1])[0]
    return policy

def calculate_avg_q_value(Q):
    if not Q:
        return 0
    total = sum(max(state_values.values()) for state_values in Q.values())
    return total / len(Q)

def calculate_policy_stability(Q):
    total_states = len(Q)
    if total_states == 0:
        return 0
    stable_states = sum(1 for state_values in Q.values() 
                       if max(state_values.values()) > 0.8 * sum(state_values.values()))
    return stable_states / total_states

def normalize_q_table(Q):
    q_table_data = {}
    for state in Q:
        values = {
            'up': float(Q[state][(-1, 0)]),
            'down': float(Q[state][(1, 0)]),
            'left': float(Q[state][(0, -1)]),
            'right': float(Q[state][(0, 1)])
        }
        # Normalize values to reasonable range
        max_abs = max(abs(v) for v in values.values())
        if max_abs > 0:
            values = {k: round(v / max_abs * 100, 2) for k, v in values.items()}
        q_table_data[str(state)] = values
    return q_table_data


# region Flask

@app.route('/')
def index():
    return render_template('index.html', map_layout=map_layout, path=[], goals=[], enumerate=enumerate)

goals_positions = []

@app.route('/set_goal', methods=['POST'])
def set_goal():
    global goals_positions
    goals_positions = [tuple(goal) for goal in request.json.get('goals', [])]
    return jsonify({"message": "Goals set", "goals": goals_positions})

@app.route('/path', methods=['GET'])
def get_path():
    algorithm = request.args.get('algorithm')
    start = (9, 0)
    global goals_positions

    if len(goals_positions) == 0:
        return jsonify({
            "path": [],
            "cost": 0,
            "performance": {
                "time": 0,
                "space": 0,
                "optimality": "No",
                "completeness": "No",
                "cost": 0
            }
        }), 400

    current_position = start
    start_time = time.time()
    final_path = []
    total_space = 0

    try:
        if algorithm == 'q_learning':
            final_path, total_space, total_reward, q_table_data, training_data, q_value_snapshots = q_learning(current_position, goals_positions)
            end_time = time.time()
            execution_time = int((end_time - start_time) * 1000)

            return jsonify({
                "path": final_path,
                "cost": len(final_path) if final_path else 0,
                "performance": {
                    "time": execution_time,
                    "space": total_space,
                    "optimality": "Yes" if final_path else "No",
                    "completeness": "Yes" if final_path else "No",
                    "cost": len(final_path) if final_path else 0,
                    "reward": total_reward
                },
                "q_table": q_table_data,
                "training_data": training_data,
                "q_value_snapshots": q_value_snapshots
            })

        
        # Original code for other algorithms
        for goal in goals_positions:
            if algorithm == 'bfs':
                path_segment, space = bfs(current_position, goal)
            elif algorithm == 'dfs':
                path_segment, space = dfs(current_position, goal)
            elif algorithm == 'ids':
                path_segment, space, cost = iterative_deepening_search(current_position, goal)
            elif algorithm == 'ucs':
                path_segment, space, cost = ucs(current_position, goal)
            elif algorithm == 'greedy_manhattan':
                path_segment, heuristics, cost, space = greedy_best_first_search(current_position, goal, heuristic_manhattan)
            elif algorithm == 'greedy_euclidean':
                path_segment, heuristics, cost, space = greedy_best_first_search(current_position, goal, heuristic_euclidean)
            elif algorithm == 'astar_euclidean':
                path_segment, heuristics, current_cost, f_value, total_cost, space = a_star_search(current_position, goal, heuristic_euclidean)
            elif algorithm == 'astar_manhattan':
                path_segment, heuristics, current_cost, f_value, total_cost, space = a_star_search(current_position, goal, heuristic_manhattan)
            elif algorithm == 'hill_climbing':
                path_segment, space = hill_climbing(current_position, goal, heuristic_manhattan)
            elif algorithm == 'simulated_annealing':
                path_segment, space = simulated_annealing(current_position, goal, heuristic_manhattan)
            elif algorithm == 'genetic_algorithms':
                path_segment, space = genetic_algorithm(current_position, goal)
            else:
                return jsonify({"error": "Algorithm not found"}), 404

            if path_segment:
                final_path.extend(path_segment if goal == goals_positions[-1] else path_segment[:-1])
                total_space += space
                current_position = goal

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    end_time = time.time()
    execution_time = int((end_time - start_time) * 1000)

    return jsonify({
        "path": final_path,
        "cost": len(final_path) if final_path else 0,
        "performance": {
            "time": execution_time,
            "space": total_space,
            "optimality": "Yes" if final_path else "No",
            "completeness": "Yes" if final_path else "No",
            "cost": len(final_path) if final_path else 0
        }
    })

@app.route('/generate-map', methods=['GET'])
def generate_map_route():
    global map_layout
    map_layout = generate_map()
    return jsonify({"map_layout": map_layout})

if __name__ == "__main__":
    socketio.run(app, debug=True)

 #endregion

