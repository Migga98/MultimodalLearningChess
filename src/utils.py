import numpy as np
import chess
import pickle
import os

NORMALIZE_MOBILITY = 64
NORMALIZE_PIECE_NUMBER = 8
NORMALIZE_50_MOVE_RULE = 50
MAX_NB_MOVES = 500

CHANNEL_PIECES = 0
CHANNEL_REPETITION = 12
CHANNEL_EN_PASSANT = 14
CHANNEL_CASTLING = 15
CHANNEL_NO_PROGRESS = 19
CHANNEL_COLOR = 20
CHANNEL_MOVE_NR = 21
CHANNEL_LAST_MOVES = 22

NB_CHANNELS_TOTAL = 24
NB_LAST_MOVES = 1


def get_row_col(position, mirror=False):
    """
    Maps a value [0,63] to its row and column index
    :param position: Position id which is an integer [0,63]
    :param mirror: Returns the indices for the mirrored board
    :return: Row and columns index
    """
    # returns the column and row index of a given position
    row = position // 8
    col = position % 8

    if mirror:
        row = 7 - row

    return row, col


def board_to_planes(board: chess.Board, board_occ=0, normalize=False, last_moves=None):
    """
    5 planes
    * * *

    Total: 22 planes
    :param board: Board handle (Python-chess object)
    :param board_occ: Number of board occurrences
    :param normalize: True if the inputs shall be normalized to the range [0.-1.]
    :param last_moves:
    :return: planes - the plane representation of the current board state
    """

    # return the plane representation of the given board
    # return variants.board_to_planes(board, board_occ, normalize, mode=MODE_CHESS)
    planes = np.zeros((24, 8, 8)).astype(int)

    # channel will be incremented by 1 at first plane
    channel = 0
    me = board.turn
    you = not board.turn
    colors = [me, you]

    # mirror all bitboard entries for the black player
    mirror = board.turn == chess.BLACK

    assert channel == CHANNEL_PIECES
    # Fill in the piece positions
    # Channel: 0 - 11
    # Iterate over both color starting with WHITE
    for color in colors:
        # the PIECE_TYPE is an integer list in python-chess
        for piece_type in chess.PIECE_TYPES:
            # iterate over the piece mask and receive every position square of it
            for pos in board.pieces(piece_type, color):
                row, col = get_row_col(pos, mirror=mirror)
                # set the bit at the right position
                planes[channel, row, col] = 1
            channel += 1

    assert channel == CHANNEL_REPETITION
    # Channel: 12 - 13
    # set how often the position has already occurred in the game (default 0 times)
    # this is used to check for claiming the 3-fold repetition rule
    if board_occ >= 1:
        planes[channel, :, :] = 1
        if board_occ >= 2:
            planes[channel + 1, :, :] = 1
    channel += 2

    # Channel: 14
    # En Passant Square
    assert channel == CHANNEL_EN_PASSANT
    if board.ep_square and board.has_legal_en_passant():  # is not None:
        row, col = get_row_col(board.ep_square, mirror=mirror)
        planes[channel, row, col] = 1
    channel += 1

    # Channel: 15 - 18
    assert channel == CHANNEL_CASTLING
    for color in colors:
        # check for King Side Castling
        if board.has_kingside_castling_rights(color):
            planes[channel, :, :] = 1
        channel += 1
        # check for Queen Side Castling
        if board.has_queenside_castling_rights(color):
            planes[channel, :, :] = 1
        channel += 1

    # Channel: 19
    # (IV.4) No Progress Count
    # define a no 'progress' counter
    # it gets incremented by 1 each move
    # however, whenever a piece gets dropped, a piece is captured or a pawn is moved, it is reset to 0
    # half-move_clock is an official metric in fen notation
    #  -> see: https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
    # check how often the position has already occurred in the game
    assert channel == CHANNEL_NO_PROGRESS
    planes[channel, :, :] = board.halfmove_clock / NORMALIZE_50_MOVE_RULE if normalize else board.halfmove_clock
    channel += 1

    assert channel == CHANNEL_COLOR
    # (IV.1) Color
    if board.turn == chess.WHITE:
        planes[channel, :, :] = 1
    # otherwise the mat will remain zero
    channel += 1

    assert channel == CHANNEL_MOVE_NR
    planes[channel, :, :] = board.fullmove_number / MAX_NB_MOVES if normalize else board.fullmove_number
    channel += 1

    # Channel: 22 - 23
    assert channel == CHANNEL_LAST_MOVES
    # Last move
    if last_moves:
        assert (len(last_moves) == NB_LAST_MOVES)
        for move in last_moves:
            if move:
                from_row, from_col = get_row_col(move.from_square, mirror=mirror)
                to_row, to_col = get_row_col(move.to_square, mirror=mirror)
                planes[channel, from_row, from_col] = 1
                channel += 1
                planes[channel, to_row, to_col] = 1
                channel += 1
            else:
                channel += 2
    else:
        channel += NB_LAST_MOVES * 2

    assert channel == NB_CHANNELS_TOTAL

    return planes


def save_as_pickle(directory, filename, data):
    completeName = os.path.join(directory, filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)


def load_pickle(directory, filename):
    completeName = os.path.join(directory, filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data
