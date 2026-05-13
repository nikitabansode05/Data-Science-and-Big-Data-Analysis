# A* (A-Star) Algorithm Implementation in Python
# Finding shortest path in a 2D grid

import heapq

# Function to calculate heuristic value (Manhattan Distance)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A* Algorithm Function
def astar(grid, start, goal):

    rows = len(grid)
    cols = len(grid[0])

    # Priority Queue
    open_list = []
    heapq.heappush(open_list, (0, start))

    # Dictionaries to store cost and path
    came_from = {}
    g_cost = {start: 0}

    while open_list:

        # Get node with lowest f(n)
        current = heapq.heappop(open_list)[1]

        # Goal reached
        if current == goal:
            path = []

            while current in came_from:
                path.append(current)
                current = came_from[current]

            path.append(start)
            path.reverse()
            return path

        # Possible movements (Up, Down, Left, Right)
        neighbors = [
            (current[0] - 1, current[1]),   # Up
            (current[0] + 1, current[1]),   # Down
            (current[0], current[1] - 1),   # Left
            (current[0], current[1] + 1)    # Right
        ]

        for neighbor in neighbors:

            r, c = neighbor

            # Check boundaries and obstacles
            if 0 <= r < rows and 0 <= c < cols and grid[r][c] == 0:

                tentative_g_cost = g_cost[current] + 1

                if neighbor not in g_cost or tentative_g_cost < g_cost[neighbor]:

                    came_from[neighbor] = current
                    g_cost[neighbor] = tentative_g_cost

                    # f(n) = g(n) + h(n)
                    f_cost = tentative_g_cost + heuristic(neighbor, goal)

                    heapq.heappush(open_list, (f_cost, neighbor))

    return None


# 0 = Free path
# 1 = Obstacle
grid = [
    [0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

start = (0, 0)
goal = (4, 4)

# Run A* Algorithm
path = astar(grid, start, goal)

# Print result
if path:
    print("Shortest Path:")
    for step in path:
        print(step, end=" ")
else:
    print("No path found")