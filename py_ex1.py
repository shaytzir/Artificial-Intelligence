import queue
class State:
    """""
    This class represents a "state" type. it keeps information
    about each state in the game. including each state's string, matrix
    father state, which direction needed to get here from father state
    in some cases will use cost for Astar for example
    """
    def __init__(self, state_arr):
        """"
        constructor. finding which indices are of '0'
        """
        self.state_arr = state_arr
        self.free_space = self.state_arr.index('0')
        self.g = None
        self.h = None
        self.move_to_get_here = None
        self.father_state = None

    @property
    def f(self):
        """"
        returns the heuristic value of this node + it's depth
        """
        return self.g + self.h


    @property
    def str(self):
        """
        return string represtation of this node
        """
        return "-".join(self.state_arr)

    def arr(self):
        """"
        returns the array of this node
        """
        return self.state_arr


    def set_father(self, father_state):
        """
        setting a father state for this current state,
        updating the required direction to get the current
        state from it's father
        """
        self.father_state = father_state

    def set_move_to_get_here(self,move):
        """
        setting which operator is required to get to this node
        """
        self.move_to_get_here = move

    def trace(self):
        """
        gets the full trace of getting from the initial state to this one
        """
        if self.father_state is None:
            return ""
        #   return it backwards, first find my
        #  father trace and add the last move
        return ""+self.father_state.trace()+self.move_to_get_here

    def __eq__(self, state):
        """
        implementing the == between states
        """
        if state is None:
            return False
        #if they share the same string
        other = state.str
        mine = self.str
        if state.str == self.str:
            return True
        return False

    def get_free_spot(self):
        """
        getter of the free space indices
        """
        return self.free_space


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
        goal_arr = [str(element) for element in range(1, self.size**2)]
        goal_arr.append('0')
        self.initial_state = State(initial_arr)
        self.goal_state = State(goal_arr)

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

    def get_mat_position(self, list_index):
        """
        returns 2D indices of a 1D index by the board size
        """
        return int(list_index / self.size), list_index % self.size

    def manhattan_dis(self,state_arr):
        """
        calcs the manhattan distance of the entire board
        :param state_arr:  the array of the node
        :return: the sum of all manhattan distances - the heuristic func
        """
        summ = 0
        i = 0
        for place in state_arr:
            if not(place=='0'):
                goal_row, goal_col = self.get_mat_position(int(place)-1)
                cur_row, cur_col = self.get_mat_position(i)
                summ += abs(goal_row - cur_row) + abs(goal_col - cur_col)
            i += 1
        return summ

    def get_possible_states(self, state):
        """
        returns all possible states from a given states
        creates the states list in this order (if exists):
        up,down,left,right
        """
        index = state.get_free_spot()
        row, col = self.get_mat_position(index)
        possible_states = []
        # gets a copy of the given state
        original = state.state_arr
        size = self.size
        blank = '0'
        # creating up to 4 different states if possible
        # by replacing the blank spot in the original state
        # with the relevant spot: upper, lower,left,right
        if row != size-1:
            up_arr = list(original)
            up_arr[index] = up_arr[index + size]
            up_arr[index + size] = blank
            up_state = State(up_arr)
            up_state.set_father(state)
            up_state.set_move_to_get_here('U')
            possible_states.append(up_state)
        if row != 0:
            down_arr = list(original)
            down_arr[index] = down_arr [index - size]
            down_arr[index - size] = blank
            down_state = State(down_arr)
            down_state.set_father(state)
            down_state.set_move_to_get_here('D')
            possible_states.append(down_state)
        if col != size-1:
            left_arr= list(original)
            left_arr[index] = left_arr[index+1]
            left_arr[index+1] = blank
            left_state = State(left_arr)
            left_state.set_father(state)
            left_state.set_move_to_get_here('L')
            possible_states.append(left_state)
        if col != 0:
            right_arr= list(original)
            right_arr[index] = right_arr[index-1]
            right_arr[index-1] = blank
            right_state = State(right_arr)
            right_state.set_father(state)
            right_state.set_move_to_get_here('R')
            possible_states.append(right_state)
        return possible_states



class Files_Manager:
    """
    The class is responsible to to read the params from the "input.txt" file
    in it's constructor it opens the file and has another method responsible
    to return all the params
    """

    def get_params(self):
        """
        reads the input file and returns the parameters to the main method
        """
        file = open('input.txt')
        algorithm = file.readline().split('\n')[0]
        size = file.readline().split('\n')[0]
        initial_state_arr = file.readline().split('\n')[0].split('-')
        file.close()
        return int(algorithm),size,initial_state_arr

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
        self.initial_state = self.board.get_initial_state()
        self.goal_state = self.board.get_goal_state()

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
        open_q = queue.Queue()
        open_q.put(self.initial_state)
        while not(open_q.empty()):
            node = open_q.get()
            self.num_of_vertexes += 1
            if node == self.goal_state:
                trace = node.trace()
                break
            successors = self.board.get_possible_states(node)
            for state in successors:
                open_q.put(state)
        return trace, self.num_of_vertexes,0


class AStar_Algorithm(Algorithm):
    """
    class of the A* algorithm. implement the search func
    finding the goal node using a heuristic func and the nodes depth
    """
    def search(self):
        """
        runs the A* algorithm as required. pops from queue the lowest f value
        if there are some with equal f - pops by order of creation and given operators order
        """
        from heapq import heappop, heappush
        state_id = 0
        open_queue = []
        initial_state = self.initial_state
        initial_state.g = 0
        initial_state.h = self.board.manhattan_dis(initial_state.arr())
        heappush(open_queue,(initial_state.f, 0, initial_state))
        state_id += 1
        while open_queue:
            state = heappop(open_queue)[2]
            print (state.str + " it's f is: " + str(state.f))
            self.num_of_vertexes += 1
            if state == self.goal_state:
                return state.trace(), self.num_of_vertexes, state.g
            successors = self.board.get_possible_states(state)
            for node in successors:
                node.g = state.g + 1
                node.h = self.board.manhattan_dis(node.arr())
                heappush(open_queue, (node.f, state_id,node))
                state_id += 1


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
        return goal.trace(), self.num_of_vertexes, limit-1

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
    """
    the main function of the code
    """
    manager = Files_Manager()
    [alg, size, arr] = manager.get_params()
    board = TileBoard(size, arr)
    if alg == 1:
        result = IDS_Algorithm(board).search()
    elif alg == 2:
        result = BFS_Algorithm(board).search()
    else:
        result = AStar_Algorithm(board).search()

    manager.print_to_file(result[0],result[1],result[2])

if __name__ == '__main__':
    main()







