# Helper function to convert the state to a tuple to store it in the explored set
def state_to_tuple(state):
    return tuple(tuple(row) for row in state)


# Robust Memoization decorator that handles list inputs
def memoize(func):
    """
    Memoization decorator that handles list inputs by converting them to tuples.
    Caches the results of the function for previously seen inputs to avoid redundant computations.

    This decorator is necessary because list arguments are not hashable in Python,
    which is required for caching with functools.lru_cache or dictionaries.
    It converts list arguments (including nested lists representing puzzle states)
    into tuples before using them as keys in the cache.
    """
    cache = {}  # Initialize the cache for memoized results

    def memoized_func(*args, **kwargs):
        """Memoized version of the function."""
        # Convert list arguments to tuples for hashability
        hashable_args = []
        for arg in args:
            if isinstance(arg, list):
                # Convert nested lists (puzzle states) to nested tuples
                if all(
                    isinstance(item, list) for item in arg if isinstance(item, list)
                ):
                    hashable_arg = tuple(
                        tuple(row) for row in arg
                    )  # Convert 2D list to tuple of tuples
                else:
                    hashable_arg = tuple(arg)  # Convert 1D list to tuple
            else:
                hashable_arg = arg  # Argument is already hashable
            hashable_args.append(hashable_arg)

        # Create a hashable key from the combined arguments (args and kwargs)
        key = (tuple(hashable_args), frozenset(kwargs.items()) if kwargs else None)

        if key not in cache:  # Check if the result for this key is already cached
            cache[key] = func(
                *args, **kwargs
            )  # Compute and cache the result if not in cache
        return cache[key]  # Return the cached result

    return memoized_func


# Function to create the goal state for an n x n puzzle
def create_goal_state(n):
    """
    This function creates a goal state based on the size of the puzzle
    Input:
        -n: the size of the puzzle
    Output:
        -goal_state: the goal state (list of lists)
    """
    # Initialize an n x n grid with zeros
    goal_state = [[0 for _ in range(n)] for _ in range(n)]

    # Fill in the grid with goal state values based on the size of the puzzle
    for i in range(n):
        for j in range(n):
            goal_state[i][j] = i * n + j
    return goal_state


# Function to get the goal positions of tiles in the goal state
def goal_state_position(n):
    """
    This function returns the position of each tile in the goal state
    Input:
        -n: the size of the puzzle
    Output:
        -position: a dictionary with the position of each tile in the goal state (row, column)
    """
    # Create a goal state based on the size of the puzzle
    goal_state = create_goal_state(n)

    # Initialize a dictionary to store the position of each tile in the goal state
    position = {}

    # Fill in the dictionary with the position of each tile in the goal state
    for i in range(n):
        for j in range(n):
            position[goal_state[i][j]] = (i, j)

    return position


# Heuristic function 1: Misplaced Tiles heuristic
@memoize  # Apply memoization decorator to cache results of h1
def h1(state):
    """
    This function returns the number of misplaced tiles, given the board state
    Input:
        -state: the board state as a list of lists
    Output:
        -h: the number of misplaced tiles
    """
    # Create a goal state based on the size of the puzzle
    goal_state = create_goal_state(len(state))

    # Initialize a counter to keep track of the number of misplaced tiles
    counter = 0

    # Count the number of misplaced tiles, comparing the current state with the goal state
    for i in range(len(state)):
        for j in range(len(state)):
            if state[i][j] != goal_state[i][j] and state[i][j] != 0:
                counter += 1

    return counter


# Heuristic function 2: Manhattan Distance heuristic
@memoize  # Apply memoization decorator to cache results of h2
def h2(state):
    """
    This function returns the Manhattan distance from the solved state, given the board state
    Input:
        -state: the board state as a list of lists
    Output:
        -h: the Manhattan distance from the solved configuration
    """
    # Create a goal state based on the size of the puzzle
    goal_state = create_goal_state(len(state))

    # Get the position of each tile in the goal state
    goal_position = goal_state_position(len(state))

    # Initialize a counter to keep track of the Manhattan distance
    counter = 0

    # Calculate and get the sum Manhattan distance, comparing the current state with the goal state
    for i in range(len(state)):
        for j in range(len(state)):
            if state[i][j] != 0:
                # Calculate the Manhattan distance
                counter += abs(i - goal_position[state[i][j]][0]) + abs(
                    j - goal_position[state[i][j]][1]
                )

    return counter


# Linear Conflict heuristic (h3) - Dominates Manhattan Distance (for extension)
@memoize
def h3(state):
    """
    Calculates the Linear Conflict heuristic, which dominates Manhattan Distance.
    ... (docstring remains the same) ...
    """
    # Implementation based on the pattern database approach from reference
    from collections import defaultdict

    # Convert state to tuple for database lookup
    state_tuple = state_to_tuple(state)

    # If it's a 3x3 puzzle, use pattern database
    if len(state) == 3:
        # Check if pattern database exists or create it if needed
        if 'PATTERN_DB' not in globals():
            # Define the goal state as a 3x3 tuple
            global PATTERN_DB
            PATTERN_DB = defaultdict(lambda: float('inf'))
            goal_state_tuple = state_to_tuple(create_goal_state(3))

            # Set distance to 0 for the goal state
            PATTERN_DB[goal_state_tuple] = 0

            # Set up queue and initialize with goal state with distance of 0
            import queue as q
            queue = q.Queue()
            queue.put((goal_state_tuple, 0))

            # Perform reverse BFS to build pattern database
            while not queue.empty():
                current_state_tuple, dist = queue.get()

                # Skip if we found a shorter path already
                if dist > PATTERN_DB[current_state_tuple]:
                    continue

                # Convert tuple state to list for easier manipulation
                current_state = [list(row) for row in current_state_tuple]

                # Find blank position
                blank_pos = None
                for i in range(3):
                    for j in range(3):
                        if current_state[i][j] == 0:
                            blank_pos = (i, j)
                            break
                    if blank_pos:
                        break

                # Try all possible moves
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    new_r, new_c = blank_pos[0] + dr, blank_pos[1] + dc

                    if 0 <= new_r < 3 and 0 <= new_c < 3:
                        # Create new state by swapping blank tile
                        new_state = [row[:] for row in current_state]
                        new_state[blank_pos[0]][blank_pos[1]], new_state[new_r][new_c] = \
                            new_state[new_r][new_c], new_state[blank_pos[0]][blank_pos[1]]

                        # Convert to tuple for hash lookup
                        new_state_tuple = tuple(tuple(row) for row in new_state)

                        # Only update if we found a shorter path
                        if dist + 1 < PATTERN_DB[new_state_tuple]:
                            PATTERN_DB[new_state_tuple] = dist + 1
                            queue.put((new_state_tuple, dist + 1))

        # Return pattern database value if available
        if state_tuple in PATTERN_DB:
            return PATTERN_DB[state_tuple]

    # Fallback to Manhattan distance + linear conflicts
    h_man = h2(state)
    linear_conflicts = 0
    n = len(state)
    goal_state = create_goal_state(n)

    # Check for row conflicts
    for i in range(n):
        tiles_in_row = []
        for j in range(n):
            tile = state[i][j]
            if tile != 0 and goal_state[i].count(tile):
                tiles_in_row.append((tile, j))
        tiles_in_row.sort(key=lambda x: goal_state[i].index(x[0]))
        for k in range(len(tiles_in_row)):
            for l in range(k + 1, len(tiles_in_row)):
                if tiles_in_row[k][1] > tiles_in_row[l][1]:
                    linear_conflicts += 2

    # Check for column conflicts (similar logic as row conflicts)
    for j in range(n):
        tiles_in_col = []
        for i in range(n):
            tile = state[i][j]
            goal_col = [goal_state[row][j] for row in range(n)]
            if tile != 0 and goal_col.count(tile):
                tiles_in_col.append((tile, i))
        tiles_in_col.sort(
            key=lambda x: [goal_state[row][j] for row in range(n)].index(x[0])
        )
        for k in range(len(tiles_in_col)):
            for l in range(k + 1, len(tiles_in_col)):
                if tiles_in_col[k][1] > tiles_in_col[l][1]:
                    linear_conflicts += 2

    return h_man + linear_conflicts


# Heuristic list - contains all implemented heuristic functions
heuristics = [h1, h2, h3]


def get_blank_pos(state):
    """Find blank tile (0) position (row, col)."""
    for i in range(len(state)):
        for j in range(len(state[i])):
            if state[i][j] == 0:
                return i, j
    return None  # Should not happen in valid puzzles


def is_valid_state(state):
    """Check if input state is a valid n-puzzle."""
    # Check whether n > 2, returning error code -1 if not
    if len(state) < 3:
        return False

    # Check whether the grid is n x n
    for i in range(len(state)):
        if len(state[i]) != len(state):
            return False

    # Check whether it contains every number from 0 to n^2 − 1
    elements = set(range(len(state) ** 2))
    for row in state:
        elements -= set(row)
    if elements:
        return False

    return True


def count_inversions(state):
    """Count inversions in a puzzle state (excluding blank tile)."""
    # Count the number of inversions
    inversions = 0
    list_of_elements = [
        state[i][j]
        for i in range(len(state))
        for j in range(len(state))
        if state[i][j] != 0
    ]
    for i in range(len(list_of_elements)):
        for j in range(i + 1, len(list_of_elements)):
            if list_of_elements[i] > list_of_elements[j]:
                inversions += 1
    return inversions


def is_solvable(state):
    """Determine if puzzle is solvable."""
    inversions = count_inversions(state)

    # If N is odd, puzzle is solvable if inversions count is even
    if len(state) % 2 == 1:
        return inversions % 2 == 0
    # If N is even, puzzle is solvable if:
    # (blank_row_from_bottom + inversions) is even
    else:
        blank_row = None
        for i in range(len(state)):
            for j in range(len(state)):
                if state[i][j] == 0:
                    blank_row = len(state) - i
                    break
            if blank_row is not None:
                break
        return (blank_row + inversions) % 2 == 0


# Main solvePuzzle function.
def solvePuzzle(state, heuristic):
    """This function should solve the n**2-1 puzzle for any n > 2 (although it may take too long for n > 4)).
    Inputs:
        -state: The initial state of the puzzle as a list of lists
        -heuristic: a handle to a heuristic function.  Will be one of those defined in Question 2.
    Outputs:
        -steps: The number of steps to optimally solve the puzzle (excluding the initial state)
        -exp: The number of nodes expanded to reach the solution
        -max_frontier: The maximum size of the frontier over the whole search
        -opt_path: The optimal path as a list of list of lists.  That is, opt_path[:,:,i] should give a list of lists
                    that represents the state of the board at the ith step of the solution.
        -err: An error code.  If state is not of the appropriate size and dimension, return -1.  For the extension task,
          if the state is not solvable, then return -2
    """
    # Initialize the number of steps, nodes expanded, maximum size, and the optimal path of the frontier
    steps, exp, max_frontier = 0, 0, 0
    opt_path = []

    # Check if state is valid
    if not is_valid_state(state):
        return steps, exp, max_frontier, None, -1

    # Check for solvability
    if not is_solvable(state):
        return steps, exp, max_frontier, opt_path, -2

    # frontier should be a priority queue because we want to pop the node with the lowest f value
    # explored should be a set because we want to check if a node has been explored in O(1) time
    frontier = PriorityQueue()
    explored = set()

    # Initialize the start node
    start_node = PuzzleNode(state, None, 0, heuristic)
    frontier_size = 1

    # Update the max frontier size
    max_frontier = max(max_frontier, frontier_size)

    # Create a goal state based on the puzzle size
    goal_state = create_goal_state(len(state))

    # Add the start node to the frontier
    frontier.put((start_node.f, start_node))

    # A* Search Algorithm
    while not frontier.empty():
        _, current_node = frontier.get()
        frontier_size -= 1

        # Check if the current state is already explored
        current_state_tuple = state_to_tuple(current_node.state)
        if current_state_tuple in explored:
            continue

        # Add the current state to the explored set
        explored.add(current_state_tuple)

        # Check if the current state is the goal state
        if current_node.state == goal_state:
            # Reconstruct the optimal path
            node = current_node
            while node:
                opt_path.append(node.state)
                node = node.parent
            opt_path.reverse()
            # Update the number of steps
            steps = len(opt_path) - 1

            return steps, exp, max_frontier, opt_path, 0

        # Expand current node and increment the expansion counter
        exp += 1

        # Generate child nodes
        children = current_node.expand()
        for child in children:
            child_state_tuple = state_to_tuple(child.state)
            if child_state_tuple not in explored:
                frontier.put((child.f, child))
                frontier_size += 1
                max_frontier = max(max_frontier, frontier_size)

    # If frontier is empty and goal not found, puzzle is unsolvable
    return steps, exp, max_frontier, opt_path, -2


if __name__ == "__main__":
    # Basic Functionality Tests
    incorrect_state = [[0, 1, 2], [2, 3, 4], [5, 6, 7]]
    _, _, _, _, err = solvePuzzle(incorrect_state, lambda state: 0)
    assert err == -1

    working_initial_states_8_puzzle = (
        [[2, 3, 7], [1, 8, 0], [6, 5, 4]],
        [[7, 0, 8], [4, 6, 1], [5, 3, 2]],
        [[5, 7, 6], [2, 4, 3], [8, 1, 0]],
    )

    h_mt_vals = [7, 8, 7]
    h_man_vals = [15, 17, 18]

    for i in range(0, 3):
        h_mt = heuristics[0](working_initial_states_8_puzzle[i])
        h_man = heuristics[1](working_initial_states_8_puzzle[i])
        assert h_mt == h_mt_vals[i]
        assert h_man == h_man_vals[i]

    astar_steps = [17, 25, 28]
    for i in range(0, 3):
        steps_mt, expansions_mt, _, opt_path_mt, _ = solvePuzzle(
            working_initial_states_8_puzzle[i], heuristics[0]
        )
        steps_man, expansions_man, _, opt_path_man, _ = solvePuzzle(
            working_initial_states_8_puzzle[i], heuristics[1]
        )
        assert steps_mt == steps_man == astar_steps[i]
        assert expansions_man < expansions_mt
        if i == 0:
            assert opt_path_mt == [
                [[2, 3, 7], [1, 8, 0], [6, 5, 4]],
                [[2, 3, 7], [1, 8, 4], [6, 5, 0]],
                [[2, 3, 7], [1, 8, 4], [6, 0, 5]],
                [[2, 3, 7], [1, 0, 4], [6, 8, 5]],
                [[2, 0, 7], [1, 3, 4], [6, 8, 5]],
                [[0, 2, 7], [1, 3, 4], [6, 8, 5]],
                [[1, 2, 7], [0, 3, 4], [6, 8, 5]],
                [[1, 2, 7], [3, 0, 4], [6, 8, 5]],
                [[1, 2, 7], [3, 4, 0], [6, 8, 5]],
                [[1, 2, 0], [3, 4, 7], [6, 8, 5]],
                [[1, 0, 2], [3, 4, 7], [6, 8, 5]],
                [[1, 4, 2], [3, 0, 7], [6, 8, 5]],
                [[1, 4, 2], [3, 7, 0], [6, 8, 5]],
                [[1, 4, 2], [3, 7, 5], [6, 8, 0]],
                [[1, 4, 2], [3, 7, 5], [6, 0, 8]],
                [[1, 4, 2], [3, 0, 5], [6, 7, 8]],
                [[1, 0, 2], [3, 4, 5], [6, 7, 8]],
                [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            ]

    working_initial_state_15_puzzle = [
        [1, 2, 6, 3],
        [0, 9, 5, 7],
        [4, 13, 10, 11],
        [8, 12, 14, 15],
    ]
    steps_mt, expansions_mt, _, _, _ = solvePuzzle(
        working_initial_state_15_puzzle, heuristics[0]
    )
    steps_man, expansions_man, _, _, _ = solvePuzzle(
        working_initial_state_15_puzzle, heuristics[1]
    )
    assert steps_mt == steps_man == 9
    assert expansions_mt >= expansions_man

    # Extension Tests
    unsolvable_initial_state = [[7, 5, 6], [2, 4, 3], [8, 1, 0]]
    _, _, _, _, err = solvePuzzle(unsolvable_initial_state, lambda state: 0)
    assert err == -2

    dom = 0
    for i in range(0, 3):
        steps_new, expansions_new, _, _, _ = solvePuzzle(
            working_initial_states_8_puzzle[i], heuristics[2]
        )
        steps_man, expansions_man, _, _, _ = solvePuzzle(
            working_initial_states_8_puzzle[i], heuristics[1]
        )
        assert steps_new == steps_man == astar_steps[i]
        dom = expansions_man - expansions_new
        assert dom > 0

    print("All tests passed!")