# N-Queens Problem using Backtracking and Branch & Bound
# Python Program

class NQueens:
    def __init__(self, n):
        self.n = n
        self.board = [["." for _ in range(n)] for _ in range(n)]

        # Arrays for Branch and Bound optimization
        self.cols = [False] * n
        self.diag1 = [False] * (2 * n - 1)   # row - col + (n-1)
        self.diag2 = [False] * (2 * n - 1)   # row + col

    # Function to print board
    def print_board(self):
        for row in self.board:
            print(" ".join(row))
        print()

    # Recursive function to solve N-Queens
    def solve(self, row):

        # Base Case: All queens placed
        if row == self.n:
            print("Solution Found:")
            self.print_board()
            return True

        # Try placing queen in each column
        for col in range(self.n):

            # Check if position is safe
            if (not self.cols[col] and
                not self.diag1[row - col + self.n - 1] and
                not self.diag2[row + col]):

                # Place Queen
                self.board[row][col] = "Q"

                # Mark column and diagonals as occupied
                self.cols[col] = True
                self.diag1[row - col + self.n - 1] = True
                self.diag2[row + col] = True

                # Recur for next row
                if self.solve(row + 1):
                    return True

                # Backtracking: Remove Queen
                self.board[row][col] = "."

                # Unmark column and diagonals
                self.cols[col] = False
                self.diag1[row - col + self.n - 1] = False
                self.diag2[row + col] = False

        return False


# Driver Code
n = 4

queens = NQueens(n)

if not queens.solve(0):
    print("No Solution Exists")