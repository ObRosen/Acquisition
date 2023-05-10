# 从checkpoint文件中读取不同代神经网络的各层形状layers
# 从网上获得黑白棋棋谱作为z^0
# 将棋谱作为自变量输入layers获得每一层神经网络的输出作为z^d

# 读取WThor文件部分代码参考：https://github.com/hjmr/wthor_reader


import struct
import argparse
from player_color import WHITE, BLACK, EMPTY
from gamemap import ReversiMap, Grid
from move_result import MoveResult
from typing import List
import pandas as pd
from load_data import into_input_format_2, dedup
import time
import random


def parse_arg():
    parser = argparse.ArgumentParser(description="WTHOR reader.")
    parser.add_argument("FILES", type=str, nargs="+", help="WTHOR files.")
    return parser.parse_args()


def unpack_common_header(bytes):
    v = struct.unpack("<4bihh4b", bytes)
    common_header = {
        "file_year_upper": v[0],
        "file_year_lower": v[1],
        "file_month": v[2],
        "file_date": v[3],
        "num_games": v[4],
        "num_record": v[5],
        "game_year": v[6],
        "board_size": v[7],
        "game_type": v[8],
        "depth": v[9],
        "reserve": v[10],
    }
    return common_header


def unpack_game_header(bytes):
    v = struct.unpack("<hhhbb", bytes)
    game_header = {
        "game_id": v[0],
        "black_player_id": v[1],
        "white_player_id": v[2],
        "black_stones": v[3],
        "black_stones_theoretical": v[4],
    }
    return game_header


def unpack_play_record(bytes):
    return struct.unpack("<60b", bytes)


def read_wthor_file(file):
    with open(file, "rb") as f:
        common_header = unpack_common_header(f.read(16))
        games = []
        for i in range(common_header["num_games"]):
            game = {}
            game["header"] = unpack_game_header(f.read(8))
            game["play"] = unpack_play_record(f.read(60))
            games.append(game)
    return (common_header, games)


def show_play_record(record):  # 这个函数读棋谱好看，实际上不需要
    num_alpha = ["a", "b", "c", "d", "e", "f", "g", "h"]
    record_str = []
    for move in record:
        upper = move // 10
        lower = move % 10
        record_str.append("{}{}".format(num_alpha[upper - 1], lower))
    print(",".join(record_str))


def getBoardState(map: ReversiMap) -> List[List[int]]:
    def _gridToState(grid: Grid) -> float:
        if grid.get() == WHITE:  # 在我们的棋盘上，1代表白棋，-1代表黑棋
            return 1
        elif grid.get() == EMPTY:
            return 0
        else:
            return -1
    return [[_gridToState(grid) for grid in row] for row in map]
    # return torch.FloatTensor([[_gridToState(grid) for grid in row] for row in map])


def show_map_state(record, random_seed: int):
    boardstates = []

    map = ReversiMap()
    map.startup_1()
    first_move = record[0]  # 看第一步确认棋盘的开局状态
    x = int(str(first_move)[0])-1
    y = int(str(first_move)[1])-1
    color = BLACK
    flag = map.gameMove(MoveResult(x, y, color))
    if flag == 2:  # 在ReversiMap中设置了报错机制。棋谱中的每一步肯定都是有效的，如果发生无棋可翻转的情况一定是开局反了，因此重新设置。
        map.startup_2()
        map.gameMove(MoveResult(x, y, color))
    boardstates.append(getBoardState(map))

    for idx, move in enumerate(record[1:]):
        if move == 0:  # move只记一个数字时应该表示游戏结束了, 直接结束for循环
            break
        x = int(str(move)[0])-1
        y = int(str(move)[1])-1
        # if x < 0 or y < 0:
        # continue
        # color = BLACK if idx % 2 else WHITE
        color = BLACK if map.isValidMove(MoveResult(x, y, BLACK)) else WHITE
        map.gameMove(MoveResult(x, y, color))
        boardstates.append(getBoardState(map))

    train_record, test_record = sampleFromSource(boardstates, random_seed)
    return train_record, test_record  # 从产生的60条8x8的boardstates中挑出这一局里采样的一个训练数据和一个测试数据


def sampleFromSource(data: List[List[List[int]]], random_seed: int):
    lst = [k for k in range(len(data))]
    random.seed(len(data)*random_seed)
    train_index = random.choice(lst)
    lst.remove(train_index)
    test_index = random.choice(lst)
    train_record = data[train_index]
    test_record = data[test_index]
    return train_record, test_record


def read_wthor_files(files):
    trainset = []
    testset = []
    game_total_num = 0
    for file in files:
        (header, games) = read_wthor_file(file)
        game_total_num += len(games)
        for idx, game in enumerate(games):
            a, b = show_map_state(game["play"], idx)
            trainset.append(a)
            testset.append(b)
    print(
        f'Total number of the othello games in the dataset: {game_total_num}')

    trset, trmemory = dedup(pd.DataFrame(trainset), set())
    #trainset_2 = into_input_format_2(trset)
    teset, _ = dedup(pd.DataFrame(testset), trmemory)
    #testset_2 = into_input_format_2(teset)
    return trset,teset
    #return trainset_2, testset_2


if __name__ == "__main__":
    paths = ['.\gamedata\WTH_' +
             str(i)+'.wtb' for i in range(1977, 2000)]  # range(1977,2024)
    start = time.time()
    trainset, testset = read_wthor_files(paths)
    middle = time.time()
    # print(middle-start)  # (1977, 2000) 全过程222s
    # print(trainset.shape)
    # print(trainset.iloc[:5, :])

    
