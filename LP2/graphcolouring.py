# Graph Coloring Problem using Backtracking and Branch & Bound
# Python Program

class GraphColoring:
    def __init__(self, vertices):
        self.V = vertices

        # Create adjacency matrix
        self.graph = [[0 for _ in range(vertices)] for _ in range(vertices)]

    # Function to check if current color assignment is safe
    def is_safe(self, vertex, color, colors):

        for i in range(self.V):

            # Check adjacent vertices
            if self.graph[vertex][i] == 1 and colors[i] == color:
                return False

        return True

    # Recursive function for graph coloring
    def solve_coloring(self, m, colors, vertex):

        # Base Case: All vertices colored
        if vertex == self.V:
            return True

        # Try all colors
        for color in range(1, m + 1):

            # Check if color can be assigned
            if self.is_safe(vertex, color, colors):

                # Assign color
                colors[vertex] = color

                # Recur for next vertex
                if self.solve_coloring(m, colors, vertex + 1):
                    return True

                # Backtracking
                colors[vertex] = 0

        return False

    # Main function
    def graph_coloring(self, m):

        colors = [0] * self.V

        if not self.solve_coloring(m, colors, 0):
            print("Solution does not exist")
            return False

        # Print solution
        print("Solution Exists:")
        for vertex in range(self.V):
            print(f"Vertex {vertex} ---> Color {colors[vertex]}")

        return True


# Driver Code
g = GraphColoring(4)

# Define graph using adjacency matrix
g.graph = [
    [0, 1, 1, 1],
    [1, 0, 1, 0],
    [1, 1, 0, 1],
    [1, 0, 1, 0]
]

# Number of colors
m = 3

# Solve Graph Coloring Problem
g.graph_coloring(m)