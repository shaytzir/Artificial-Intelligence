from Queue import Queue, PriorityQueue
from copy import deepcopy


class State:
    """""
    This class represents a "state" type. it keeps information
    about each state in the game. including each state's string, matrix
    father state, which direction needed to get here from father state
    in some cases will use cost for Astar for example
    """
    def __init__(self, state):
        """"
        constructor. finding which indices are of '0'
        """
        self.mat_state = state
        self.str_state = ""
        self.size = int(len(self.mat_state))
        self.free_row=0
        self.free_col=0
        for i in range(self.size):
            for j in range(self.size):
                element = self.mat_state[i][j]
                self.str_state += element
                if element=='0':
                    self.free_row = i
                    self.free_col = j
        self.cost = None
        self.move_to_get_here = None
        self.father_state = None

    def set_father(self, father_state):
        """"
        setting a father state for this current state,
        updating the required direction to get the current
        state from it's father
        """
        self.father_state = father_state
        [father_row,father_col] = father_state.get_free_spot()
        [row_diff, col_diff] = [father_row - self.free_row,father_col - self.free_col]
        if row_diff == -1:
            self.move_to_get_here = 'U'
        elif row_diff == 1:
            self.move_to_get_here = 'D'
        elif col_diff == -1:
            self.move_to_get_here = 'L'
        else:
            self.move_to_get_here = 'R'

    def set_cost(self, cost):
        """"
        setter. the cost of this state
        """
        self.cost = cost

    def get_state_str(self):
        """
        getter of this state's string. returns the matrix values in a row
        """
        return self.str_state

    def get_state_mat(self):
        """
        returns a copy of this state's matrix
        """
        return deepcopy(self.mat_state)

    def __eq__(self, state):
        """
        implementing the == between states
        """
        if state is None:
            return False
        #if they share the same string
        if state.get_state_str() == self.get_state_str():
            return True
        return False

    def get_free_spot(self):
        """
        getter of the free space indices
        """
        return [self.free_row, self.free_col]

    def get_trace(self):
        """
        gets the full trace of getting from the initial state to this one
        """
        if self.father_state is None:
            return ""
        #   return it backwards, first find my
        #  father trace and add the last move
        return ""+self.father_state.get_trace()+self.move_to_get_here


class TileBoard:
    """
    the class represents the game itself, keeping the initial and goal states
    able to find the next possible states from a given state
    """
    def __init__(self, board_size, initial_arr):
        """
        constructor
        creates an initial state out of the given array, and a goal state
         out of the given board size
        """
        self.size = int(board_size)
        initial_mat = [[0 for x in range(self.size)] for y in range(self.size)]
        goal_mat = [[0 for x in range(self.size)] for y in range(self.size)]
        index = 0
        for i in range(self.size):
            for j in range(self.size):
                initial_mat[i][j] = initial_arr[index]
                index += 1
                goal_mat[i][j] = str(index)
        goal_mat[self.size-1][self.size-1] = str(0)
        self.initial_state = State(initial_mat)
        self.goal_state = State(goal_mat)

    def get_initial_state(self):
        """
        getter of the initial state
        """
        return self.initial_state

    def get_goal_state(self):
        """
        getter of the goal state
        """
        return self.goal_state


    def get_possible_states(self, state):
        """
        returns all possible states from a given states
        creates the states list in this order (if exists):
        up,down,left,right
        """
        row, col = state.get_free_spot()
        row = int(row)
        col = int(col)
        possible_states = []
        # gets a copy of the given state
        original = state.get_state_mat()
        up = deepcopy(original)
        down = deepcopy(original)
        left = deepcopy(original)
        right = deepcopy(original)
        size = len(original)
        blank = '0'
        # creating up to 4 different states if possible
        # by replacing the blank spot in the original state
        # with the relevant spot: upper, lower,left,right
        if row != size-1:
            up[row][col] = original[row+1][col]
            up[row + 1][col] = blank
            up_state = State(up)
            possible_states.append(up_state)
        if row != 0:
            down[row][col] = original[row-1][col]
            down[row - 1][col] = blank
            down_state = State(down)
            possible_states.append(down_state)
        if col != size-1:
            left[row][col] = original[row][col+1]
            left[row][col + 1] = blank
            left_state = State(left)
            possible_states.append(left_state)
        if col != 0:
            right[row][col] = original[row][col-1]
            right[row][col-1] = blank
            right_state = State(right)
            possible_states.append(right_state)
        return possible_states



class Files_Manager:
    """
    The class is responsible to to read the params from the "input.txt" file
    in it's constructor it opens the file and has another method responsible
    to return all the params
    """
    def __init__(self):
        """
        opens the input file
        """
        self.file = open('input.txt')
        self.size = None
        self.algorithm = None
        self.initial_state_arr = None

    def get_params(self):
        """
        reads the input file and returns the parameters to the main method
        """
        self.algorithm = self.file.readline().split('\n')[0]
        self.size = self.file.readline().split('\n')[0]
        self.initial_state_arr = self.file.readline().split('\n')[0].split('-')
        self.file.close()
        return int(self.algorithm),self.size,self.initial_state_arr

    def print_to_file(self, trace, vertex_num, third_param):
        """
        writes to the "output.txt" file the information about the search:
        the trace in order to get to the goal, number of developed vertexes,
        and a seperate third solution for each alg
        """
        f = open('output.txt','w')
        f.write(trace + " " + str(vertex_num) + " " + str(third_param))
        f.close()


class Algorithm(object):
    """
    the class is super class to all types of algorithms
    it holds two properties: developed vertexes (initialized to 0)
    and a given tile game board. all algorithms based on this class
    """
    def __init__(self, board):
        """
        initialize properties - number of developed vertexes and the board
        """
        self.num_of_vertexes = 0
        self.board = board

    def search(self):
        """
        each specific algorithm should implement it by it's own rules
        """
        pass



class BFS_Algorithm(Algorithm):
    """
    class is responsible to implement the search method of the BFS algorithm
    at the at returns the trace and the number of developed vertexes
    """
    def search(self):
        """
        implements the bfs search algorithm using queue
        """
        open_q = Queue()
        open_q.put(self.board.get_initial_state())
        while not(open_q.empty()):
            node = open_q.get()
            self.num_of_vertexes += 1
            if node == self.board.get_goal_state():
                trace = node.get_trace()
                break
            successors = self.board.get_possible_states(node)
            for state in successors:
                state.set_father(node)
                open_q.put(state)
        return trace, self.num_of_vertexes,0


class IDS_Algorithm(Algorithm):
    """
    implements the ids algorithm. the search function calls for a
    dfs search that has a limitation on the depth of the seach
    """
    def search(self):
        """
        initials the limit to be 0 and calls the limit dfs
        if a goal isn't found, the limits get bigger by 1 and
        the limit dfs is called again.
        """
        limit = 0
        goal = None
        while not goal:
            node_to_start_from = self.board.get_initial_state()
            self.num_of_vertexes = 0
            goal = self.DFS_With_Limit(node_to_start_from, limit, self.num_of_vertexes)
            limit += 1
        return goal.get_trace(), self.num_of_vertexes, limit-1

    def DFS_With_Limit(self, start, limit, nodes_developed):
        """"
        running a dfs search with a limitation on the search depth
        calling a dfs limit search of each state's child with a limitation
        smaller by one
        """
        self.num_of_vertexes += 1
        if start == self.board.get_goal_state():
            return start
        # can develop current state children
        if limit > 0:
            stack = self.board.get_possible_states(start)
            for move in stack:
                move.set_father(start)
                goal = self.DFS_With_Limit(move, limit - 1, nodes_developed)
                if goal:
                    return goal
        return None



def main():
    manager = Files_Manager()
    [alg, size, arr] = manager.get_params()
    board = TileBoard(size, arr)
    if alg == 1:
        result = IDS_Algorithm(board).search()
    elif alg == 2:
        result = BFS_Algorithm(board).search()
    else:
        print "should run Astar"

    manager.print_to_file(result[0],result[1],result[2])

if __name__ == '__main__':
    main()







