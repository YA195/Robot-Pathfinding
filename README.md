![Screenshot 2024-11-17 204639](https://github.com/user-attachments/assets/b7bec409-153d-420d-a9d2-0e63c70078a1)

# Robot Pathfinding and Algorithm Comparison

This project focuses on comparing different search algorithms for robot pathfinding and maze solving. It includes multiple search strategies to find the optimal path for a robot in a maze-like environment. The user can generate random maps with constraints, select a goal, and run various algorithms to compare their performance in terms of time, space, optimality, completeness, and cost.

## Technologies Used

- **Python**: Backend logic and algorithm implementations.
- **HTML**: Structure and layout of the web interface.
- **CSS**: Styling and design of the web pages.
- **JavaScript**: Interactivity and client-side functionality for map generation, goal selection, and algorithm execution.

## Features

- **Map Generation**: Automatically generate random maps with configurable constraints.
- **Goal Selection**: The user can select the goal point directly from the map.
- **Algorithm Comparison**: Compare different pathfinding algorithms based on:
  - Time (in microseconds)
  - Space (memory usage)
  - Optimality (whether the algorithm returns the optimal path)
  - Completeness (whether the algorithm finds a path)
  - Cost (number of steps in the path)

## Supported Algorithms

### Uninformed Search:
- **Breadth-First Search (BFS)**: Explores all possible paths evenly.
- **Depth-First Search (DFS)**: Explores as far down a branch as possible before backtracking.
- **Uniform Cost Search (UCS)**: Finds the lowest cost path by expanding the least costly nodes first.
- **Iterative Deepening Search (IDS)**: Combines the depth-first approach with breadth-first principles.

### Heuristic Search:
- **Greedy Best-First Search (Euclidean)**: A heuristic-driven search that uses Euclidean distance to estimate proximity to the goal.
- **Greedy Best-First Search (Manhattan)**: A heuristic-driven search that uses Manhattan distance to estimate proximity to the goal.
- **A* Search (Euclidean)**: Combines the benefits of BFS and Greedy Best-First using both path cost and heuristic.
- **A* Search (Manhattan)**: Similar to A* with Manhattan distance as the heuristic.

### Local Search:
- **Hill Climbing**: Continuously moves towards the steepest ascent until it reaches a peak.
- **Simulated Annealing**: A probabilistic search that avoids getting stuck in local maxima by allowing occasional downhill moves.
- **Genetic Algorithm**: A population-based search algorithm inspired by natural selection and genetics.

## Usage

### Map Generation:
- Generate a random map with a specified size.
- Add walls and obstacles to the map to simulate a maze environment.
  
### Goal Selection:
- The user can select the goal point from the map by clicking on the grid.

### Algorithm Execution:
- Select an algorithm from the list of available search algorithms.
- The selected algorithm will attempt to find a path from a start point to the goal and provide the performance metrics.

### Performance Metrics:
- **Time (µs)**: The total time taken by the algorithm to find the path.
- **Space**: The amount of memory consumed by the algorithm.
- **Optimality**: Whether the algorithm returns the optimal path (Yes/No).
- **Completeness**: Whether the algorithm finds a path (Yes/No).
- **Cost**: The number of steps taken in the path.

## Installation

To run this project locally, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/robot-pathfinding.git
   cd robot-pathfinding

