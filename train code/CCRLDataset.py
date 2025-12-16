import chess.pgn
import numpy as np
import os
import torch
from torch.utils.data import Dataset

# FIX: Import encoder from the AlphaLite package
from AlphaLite import encoder


def tolist(mainline_moves):
    """Convert an iterable of moves to a list."""
    return list(mainline_moves)


class CCRLDataset(Dataset):
    """
    PyTorch Dataset for CCRL games.
    Scans directories recursively for .pgn files.
    """

    def __init__(self, ccrl_dir):
        self.ccrl_dir = ccrl_dir
        self.pgn_file_paths = []

        print(f"[Dataset] Scanning files recursively in: {ccrl_dir}")
        for root, _, files in os.walk(ccrl_dir):
            for file in files:
                if file.endswith(".pgn"):
                    self.pgn_file_paths.append(os.path.join(root, file))

        print(f"[Dataset] Found {len(self.pgn_file_paths)} files.")

    def __len__(self):
        return len(self.pgn_file_paths)

    def __getitem__(self, idx):
        pgn_file_path = self.pgn_file_paths[idx]

        try:
            with open(pgn_file_path) as pgn_fh:
                game = chess.pgn.read_game(pgn_fh)

            if game is None:
                raise ValueError(f"Empty or corrupt game file: {pgn_file_path}")

            moves = tolist(game.mainline_moves())
            if len(moves) < 1:
                # Retry with the next index if game has no moves
                return self.__getitem__((idx + 1) % len(self))

            # Select a random position from the game
            rand_idx = int(np.random.random() * (len(moves) - 1))
            board = game.board()

            next_move = None
            for i, move in enumerate(moves):
                board.push(move)
                if i == rand_idx:
                    next_move = moves[i + 1]
                    break

            winner = encoder.parseResult(game.headers['Result'])
            position, policy, value, mask = encoder.encodeTrainingPoint(board, next_move, winner)

            return {
                'position': torch.from_numpy(position),
                'policy': torch.tensor([policy], dtype=torch.long),
                'value': torch.tensor([value], dtype=torch.float32),
                'mask': torch.from_numpy(mask)
            }

        except Exception as e:
            # print(f"Warning: Skipping {pgn_file_path} due to error: {e}")
            # Robustness: Try next file on error
            return self.__getitem__((idx + 1) % len(self))
