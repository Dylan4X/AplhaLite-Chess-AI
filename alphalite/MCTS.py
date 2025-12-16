# 注意：这里改成了 from . import encoder
from . import encoder
import math
from threading import Thread
import time

def calcUCT(edge, N_p):
    """
    Calculate the UCT formula.
    """
    Q = edge.getQ()
    N_c = edge.getN()
    P = edge.getP()
    C = 1.5
    UCT = Q + P * C * math.sqrt(N_p) / (1 + N_c)
    assert not math.isnan(UCT), 'Q {} N_c {} P {}'.format(Q, N_c, P)
    return UCT

class Node:
    def __init__(self, board, new_Q, move_probabilities):
        self.N = 1.
        self.sum_Q = new_Q
        self.edges = []
        for idx, move in enumerate(board.legal_moves):
            edge = Edge(move, move_probabilities[idx])
            self.edges.append(edge)

    def getN(self):
        return self.N

    def getQ(self):
        return self.sum_Q / self.N

    def UCTSelect(self):
        max_uct = -1000.
        max_edge = None
        for edge in self.edges:
            uct = calcUCT(edge, self.N)
            if max_uct < uct:
                max_uct = uct
                max_edge = edge
        assert not (max_edge == None and not self.isTerminal())
        return max_edge

    def maxNSelect(self):
        max_N = -1
        max_edge = None
        for edge in self.edges:
            N = edge.getN()
            if max_N < N:
                max_N = N
                max_edge = edge
        return max_edge

    def getStatisticsString(self):
        string = '|{: ^10}|{: ^10}|{: ^10}|{: ^10}|{: ^10}|\n'.format(
            'move', 'P', 'N', 'Q', 'UCT')
        edges = self.edges.copy()
        edges.sort(key=lambda edge: edge.getN())
        edges.reverse()
        for edge in edges:
            move = edge.getMove()
            P = edge.getP()
            N = edge.getN()
            Q = edge.getQ()
            UCT = calcUCT(edge, self.N)
            string += '|{: ^10}|{:10.4f}|{:10.4f}|{:10.4f}|{:10.4f}|\n'.format(
                str(move), P, N, Q, UCT)
        return string

    def isTerminal(self):
        return len(self.edges) == 0

class Edge:
    def __init__(self, move, move_probability):
        self.move = move
        self.P = move_probability
        self.child = None
        self.virtualLosses = 0.

    def has_child(self):
        return self.child != None

    def getN(self):
        if self.has_child():
            return self.child.N + self.virtualLosses
        else:
            return 0. + self.virtualLosses

    def getQ(self):
        if self.has_child():
            return 1. - ((self.child.sum_Q + self.virtualLosses) / (self.child.N + self.virtualLosses))
        else:
            return 0.

    def getP(self):
        return self.P

    def expand(self, board, new_Q, move_probabilities):
        if self.child == None:
            self.child = Node(board, new_Q, move_probabilities)
            return True
        else:
            return False

    def getChild(self):
        return self.child

    def getMove(self):
        return self.move

    def addVirtualLoss(self):
        self.virtualLosses += 1

    def clearVirtualLoss(self):
        self.virtualLosses = 0.

class Root(Node):
    def __init__(self, board, neuralNetwork):
        value, move_probabilities = encoder.callNeuralNetwork(board, neuralNetwork)
        Q = value / 2. + 0.5
        super().__init__(board, Q, move_probabilities)
        self.same_paths = 0

    def selectTask(self, board, node_path, edge_path):
        cNode = self
        while True:
            node_path.append(cNode)
            cEdge = cNode.UCTSelect()
            edge_path.append(cEdge)
            if cEdge == None:
                assert cNode.isTerminal()
                break
            cEdge.addVirtualLoss()
            board.push(cEdge.getMove())
            if not cEdge.has_child():
                break
            cNode = cEdge.getChild()

    def rollout(self, board, neuralNetwork):
        node_path = []
        edge_path = []
        self.selectTask(board, node_path, edge_path)
        edge = edge_path[-1]

        if edge != None:
            value, move_probabilities = encoder.callNeuralNetwork(board, neuralNetwork)
            new_Q = value / 2. + 0.5
            edge.expand(board, new_Q, move_probabilities)
            new_Q = 1. - new_Q
        else:
            winner = encoder.parseResult(board.result())
            if not board.turn:
                winner *= -1
            new_Q = float(winner) / 2. + 0.5

        last_node_idx = len(node_path) - 1
        for i in range(last_node_idx, -1, -1):
            node = node_path[i]
            node.N += 1
            if (last_node_idx - i) % 2 == 0:
                node.sum_Q += new_Q
            else:
                node.sum_Q += 1. - new_Q

        for edge in edge_path:
            if edge != None:
                edge.clearVirtualLoss()

    def parallelRollouts(self, board, neuralNetwork, num_parallel_rollouts):
        boards = []
        node_paths = []
        edge_paths = []
        threads = []
        for i in range(num_parallel_rollouts):
            boards.append(board.copy())
            node_paths.append([])
            edge_paths.append([])
            threads.append(Thread(target=self.selectTask,
                                  args=(boards[i], node_paths[i], edge_paths[i])))
            threads[i].start()
            time.sleep(0.0001)

        for i in range(num_parallel_rollouts):
            threads[i].join()

        values, move_probabilities = encoder.callNeuralNetworkBatched(boards, neuralNetwork)

        for i in range(num_parallel_rollouts):
            edge = edge_paths[i][-1]
            board = boards[i]
            value = values[i]
            if edge != None:
                new_Q = value / 2. + 0.5
                isunexpanded = edge.expand(board, new_Q, move_probabilities[i])
                if not isunexpanded:
                    self.same_paths += 1
                new_Q = 1. - new_Q
            else:
                winner = encoder.parseResult(board.result())
                if not board.turn:
                    winner *= -1
                new_Q = float(winner) / 2. + 0.5

            last_node_idx = len(node_paths[i]) - 1
            for r in range(last_node_idx, -1, -1):
                node = node_paths[i][r]
                node.N += 1.
                if (last_node_idx - r) % 2 == 0:
                    node.sum_Q += new_Q
                else:
                    node.sum_Q += 1. - new_Q

            for edge in edge_paths[i]:
                if edge != None:
                    edge.clearVirtualLoss()


