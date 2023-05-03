# 下列概念函数都属于多值离散函数，for all other concepts, we use linear regression


def get_movables(board, current):
    # 对于一个方向是否可以放置, 比如向右边是 dy = 0, dx = 1 的情况
    def get_movable_by_step(i, j, dy, dx):
        result = []
        isEnd = False
        while True:
            i += dy
            j += dx
            if 0 > i or i > 7 or 0 > j or j > 7 or board[i][j] == 0:
                break
            elif board[i][j] == -current:
                result.append((i, j))
                isEnd = True
            elif board[i][j] == current and isEnd == False:
                break
            elif board[i][j] == current and isEnd == True:
                return result
        return []

    # 八个不同的方向是否可行
    def get_movable_for_all_direction(i, j):
        result = []
        for dy, dx in [(dy, dx) for dy in range(-1, 2) for dx in range(-1, 2) if dy != 0 or dx != 0]:
            result += get_movable_by_step(i, j, dy, dx)
        return result

    # 对棋盘的每一个空格进行判断
    result = {}
    for i in range(8):
        for j in range(8):
            lst = get_movable_for_all_direction(i, j)
            if board[i][j] == 0 and len(lst) > 0:
                result[(i, j)] = lst
    return result

def mobility(board, current):  # 行动力
    mobility = len(get_movables(board, current).keys())
    return mobility

def frontier(board, current):  # 前沿子
    frontier = 0

    def is_frontier(i, j):
        for dy, dx in [(dy, dx) for dy in range(-1, 2) for dx in range(-1, 2) if dy != 0 or dx != 0]:
            if board[i + dy][j + dx] != 0:
                return True
        return False

    for i in range(1, 7):
        for j in range(1, 7):
            if not board[i][j] == 0 and is_frontier(i, j):
                frontier -= board[i][j]*current

    return frontier

def corner_steady(board, current):
    corner = 0
    steady = 0
    corner_map = [
        # 角点 i, j, 偏移方向 dy, dx
        [0, 0, 1, 1],
        [0, 7, 1, -1],
        [7, 0, -1, 1],
        [7, 7, -1, -1]
    ]
    for corner_i, corner_j, dy, dx in corner_map:
        if board[corner_i][corner_j] == 0:
            # 角点为空时, 如果下了临角点或对角点, 这些点很危险
            corner += board[corner_i][corner_j +
                                        dx] * current * -3
            corner += board[corner_i +
                            dy][corner_j] * current * -3
            corner += board[corner_i +
                            dy][corner_j + dx] * current * -6
            # 角点为空时, 如果下了隔角点, 这些点很好
            corner += board[corner_i][corner_j +
                                        2 * dx] * current * 4
            corner += board[corner_i + 2 *
                            dy][corner_j] * current * 4
            corner += board[corner_i +
                            dy][corner_j + 2 * dx] * current * 2
            corner += board[corner_i + 2 *
                            dy][corner_j + dx] * current * 2
        else:
            i, j = corner_i, corner_j
            # 角点的权值
            corner += board[corner_i][corner_j] * \
                current * 15
            # 角点不为空时, 处理稳定子, 为了简化运算, 仅仅考虑边稳定子
            current_color = board[corner_i][corner_j]
            while 0 <= i <= 7 and board[i][corner_j] == current_color:
                steady += current * current_color
                i += dy
            while 0 <= j <= 7 and board[corner_i][j] == current_color:
                steady += current * current_color
                j += dx
    return corner, steady

def corner(board, current):
    corner, steady = corner_steady(board, current)
    return corner

def steady(board, current):
    corner, steady = corner_steady(board, current)
    return steady

def score(board, current):  # score只是上述项的线性组合，再找一次线性回归似乎没有意义
    corner, steady = corner_steady(board, current)
    mobility = mobility(board, current)
    frontier = frontier(board, current)
    return 8 * corner + 12 * steady + 8 * mobility + 4 * frontier

def mobility_diff(board, current):
    mobi_current = mobility(board, current)
    rival = -current
    mobi_rival = mobility(board, rival)
    return mobi_current-mobi_rival

def frontier_diff(board, current):
    fron_current = frontier(board, current)
    rival = -current
    fron_rival = frontier(board, rival)
    return fron_current-fron_rival

def steady_diff(board, current):
    steady_current = steady(board, current)
    rival = -current
    steady_rival = steady(board, rival)
    return steady_current-steady_rival

def corner_diff(board, current):
    corner_current = corner(board, current)
    rival = -current
    corner_rival = corner(board, rival)
    return corner_current-corner_rival

def score_f1(board, current):
    return 4 * mobility_diff(board, current) + steady_diff(board, current)
