import chess
import numpy as np
import torch

# 自动检测 CUDA
cuda = torch.cuda.is_available()


def parseResult(result):
    if result == "1-0":
        return 1
    elif result == "1/2-1/2":
        return 0
    elif result == "0-1":
        return -1
    else:
        raise Exception("Unexpected result string {}. Exiting".format(result))


def encodePosition(board):
    planes = np.zeros((16, 8, 8), dtype=np.float32)
    wPawns = board.pieces(chess.PAWN, chess.WHITE)
    wPawns = [(chess.square_rank(sq), chess.square_file(sq)) for sq in wPawns]
    for r, f in wPawns:
        planes[0, r, f] = 1.
    bPawns = board.pieces(chess.PAWN, chess.BLACK)
    bPawns = [(chess.square_rank(sq), chess.square_file(sq)) for sq in bPawns]
    for r, f in bPawns:
        planes[1, r, f] = 1.
    wRooks = board.pieces(chess.ROOK, chess.WHITE)
    wRooks = [(chess.square_rank(sq), chess.square_file(sq)) for sq in wRooks]
    for r, f in wRooks:
        planes[2, r, f] = 1.
    bRooks = board.pieces(chess.ROOK, chess.BLACK)
    bRooks = [(chess.square_rank(sq), chess.square_file(sq)) for sq in bRooks]
    for r, f in bRooks:
        planes[3, r, f] = 1.
    wBishops = board.pieces(chess.BISHOP, chess.WHITE)
    wBishops = [(chess.square_rank(sq), chess.square_file(sq)) for sq in wBishops]
    for r, f in wBishops:
        planes[4, r, f] = 1.
    bBishops = board.pieces(chess.BISHOP, chess.BLACK)
    bBishops = [(chess.square_rank(sq), chess.square_file(sq)) for sq in bBishops]
    for r, f in bBishops:
        planes[5, r, f] = 1.
    wKnights = board.pieces(chess.KNIGHT, chess.WHITE)
    wKnights = [(chess.square_rank(sq), chess.square_file(sq)) for sq in wKnights]
    for r, f in wKnights:
        planes[6, r, f] = 1.
    bKnights = board.pieces(chess.KNIGHT, chess.BLACK)
    bKnights = [(chess.square_rank(sq), chess.square_file(sq)) for sq in bKnights]
    for r, f in bKnights:
        planes[7, r, f] = 1.
    wQueens = board.pieces(chess.QUEEN, chess.WHITE)
    wQueens = [(chess.square_rank(sq), chess.square_file(sq)) for sq in wQueens]
    for r, f in wQueens:
        planes[8, r, f] = 1.
    bQueens = board.pieces(chess.QUEEN, chess.BLACK)
    bQueens = [(chess.square_rank(sq), chess.square_file(sq)) for sq in bQueens]
    for r, f in bQueens:
        planes[9, r, f] = 1.
    wKings = board.pieces(chess.KING, chess.WHITE)
    wKings = [(chess.square_rank(sq), chess.square_file(sq)) for sq in wKings]
    for r, f in wKings:
        planes[10, r, f] = 1.
    bKings = board.pieces(chess.KING, chess.BLACK)
    bKings = [(chess.square_rank(sq), chess.square_file(sq)) for sq in bKings]
    for r, f in bKings:
        planes[11, r, f] = 1.
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[12, :, :] = 1.
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[13, :, :] = 1.
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[14, :, :] = 1.
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.
    return planes


def moveToIdx(move):
    from_rank = chess.square_rank(move.from_square)
    from_file = chess.square_file(move.from_square)
    to_rank = chess.square_rank(move.to_square)
    to_file = chess.square_file(move.to_square)

    if from_rank == to_rank and from_file < to_file:
        directionPlane = 0
        distance = to_file - from_file
        directionAndDistancePlane = directionPlane + distance
    elif from_rank == to_rank and from_file > to_file:
        directionPlane = 8
        distance = from_file - to_file
        directionAndDistancePlane = directionPlane + distance
    elif from_file == to_file and from_rank < to_rank:
        directionPlane = 16
        distance = to_rank - from_rank
        directionAndDistancePlane = directionPlane + distance
    elif from_file == to_file and from_rank > to_rank:
        directionPlane = 24
        distance = from_rank - to_rank
        directionAndDistancePlane = directionPlane + distance
    elif to_file - from_file == to_rank - from_rank and to_file - from_file > 0:
        directionPlane = 32
        distance = to_rank - from_rank
        directionAndDistancePlane = directionPlane + distance
    elif to_file - from_file == to_rank - from_rank and to_file - from_file < 0:
        directionPlane = 40
        distance = from_rank - to_rank
        directionAndDistancePlane = directionPlane + distance
    elif to_file - from_file == -(to_rank - from_rank) and to_file - from_file > 0:
        directionPlane = 48
        distance = to_file - from_file
        directionAndDistancePlane = directionPlane + distance
    elif to_file - from_file == -(to_rank - from_rank) and to_file - from_file < 0:
        directionPlane = 56
        distance = from_file - to_file
        directionAndDistancePlane = directionPlane + distance
    elif to_file - from_file == 1 and to_rank - from_rank == 2:
        directionAndDistancePlane = 64
    elif to_file - from_file == 2 and to_rank - from_rank == 1:
        directionAndDistancePlane = 65
    elif to_file - from_file == 2 and to_rank - from_rank == -1:
        directionAndDistancePlane = 66
    elif to_file - from_file == 1 and to_rank - from_rank == -2:
        directionAndDistancePlane = 67
    elif to_file - from_file == -1 and to_rank - from_rank == 2:
        directionAndDistancePlane = 68
    elif to_file - from_file == -2 and to_rank - from_rank == 1:
        directionAndDistancePlane = 69
    elif to_file - from_file == -2 and to_rank - from_rank == -1:
        directionAndDistancePlane = 70
    elif to_file - from_file == -1 and to_rank - from_rank == -2:
        directionAndDistancePlane = 71
    return directionAndDistancePlane, from_rank, from_file


def getLegalMoveMask(board):
    mask = np.zeros((72, 8, 8), dtype=np.int32)
    for move in board.legal_moves:
        planeIdx, rankIdx, fileIdx = moveToIdx(move)
        mask[planeIdx, rankIdx, fileIdx] = 1
    return mask


def mirrorMove(move):
    from_square = move.from_square
    to_square = move.to_square
    new_from_square = chess.square_mirror(from_square)
    new_to_square = chess.square_mirror(to_square)
    return chess.Move(new_from_square, new_to_square)


def encodeTrainingPoint(board, move, winner):
    if not board.turn:
        board = board.mirror()
        winner *= -1
        move = mirrorMove(move)
    positionPlanes = encodePosition(board)
    planeIdx, rankIdx, fileIdx = moveToIdx(move)
    moveIdx = planeIdx * 64 + rankIdx * 8 + fileIdx
    mask = getLegalMoveMask(board)
    return positionPlanes, moveIdx, float(winner), mask


def encodePositionForInference(board):
    if not board.turn:
        board = board.mirror()
    positionPlanes = encodePosition(board)
    mask = getLegalMoveMask(board)
    return positionPlanes, mask


def decodePolicyOutput(board, policy):
    move_probabilities = np.zeros(200, dtype=np.float32)
    num_moves = 0
    for idx, move in enumerate(board.legal_moves):
        if not board.turn:
            move = mirrorMove(move)
        planeIdx, rankIdx, fileIdx = moveToIdx(move)
        moveIdx = planeIdx * 64 + rankIdx * 8 + fileIdx
        move_probabilities[idx] = policy[moveIdx]
        num_moves += 1
    return move_probabilities[:num_moves]


def callNeuralNetwork(board, neuralNetwork):
    position, mask = encodePositionForInference(board)
    position = torch.from_numpy(position)[None, ...]
    mask = torch.from_numpy(mask)[None, ...]
    if cuda:
        position = position.cuda()
        mask = mask.cuda()
    value, policy = neuralNetwork(position, policyMask=mask)

    # === 修复：增加 detach() ===
    value = value.detach().cpu().numpy()[0, 0]
    policy = policy.detach().cpu().numpy()[0]

    move_probabilities = decodePolicyOutput(board, policy)
    return value, move_probabilities


def callNeuralNetworkBatched(boards, neuralNetwork):
    num_inputs = len(boards)
    inputs = torch.zeros((num_inputs, 16, 8, 8), dtype=torch.float32)
    masks = torch.zeros((num_inputs, 72, 8, 8), dtype=torch.float32)
    for i in range(num_inputs):
        position, mask = encodePositionForInference(boards[i])
        inputs[i] = torch.from_numpy(position)
        masks[i] = torch.from_numpy(mask)
    if cuda:
        inputs = inputs.cuda()
        masks = masks.cuda()
    value, policy = neuralNetwork(inputs, policyMask=masks)

    move_probabilities = np.zeros((num_inputs, 200), dtype=np.float32)

    # === 修复：增加 detach() ===
    value = value.detach().cpu().numpy().reshape((num_inputs))
    policy = policy.detach().cpu().numpy()

    for i in range(num_inputs):
        move_probabilities_tmp = decodePolicyOutput(boards[i], policy[i])
        move_probabilities[i, : move_probabilities_tmp.shape[0]] = move_probabilities_tmp
    return value, move_probabilities

