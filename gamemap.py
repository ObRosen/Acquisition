from typing import List, Tuple

from move_result import MoveResult
from player_color import BLACK, EMPTY, WHITE


class Grid(object):
    """
    表示棋盘的一个格子.
    """
    __slots__ = ["__type"]

    def __init__(self) -> None:
        self.__type = EMPTY

    def setColor(self, color: int):
        if self.__type != EMPTY:
            raise RuntimeError("已经占有的格子")
        if color != WHITE and color != BLACK:
            raise RuntimeError("无效的颜色")
        self.__type = color

    def reverse(self):
        if self.__type == EMPTY:
            raise RuntimeError("无法翻转没有棋子的格子")
        if self.__type == WHITE:
            self.__type = BLACK
        else:
            self.__type = WHITE

    def get(self):
        return self.__type


class ReversiMap(object):
    """
    表示一个棋盘.
    """

    ReversiMapDirections: List[Tuple[int, int]] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1)
    ]

    def __init__(self) -> None:
        self.clean()

    def clean(self):
        """清空整个棋盘."""
        self.__map = [[Grid() for _ in range(8)] for _ in range(8)]
        self.__count = 0

    @property
    def count(self) -> int:
        """返回当前棋盘上的棋子数."""
        return self.__count

    def startup_1(self):
        """放置中间四个初始棋子."""
        self.clean()
        self.getGrid(3, 3).setColor(WHITE)
        self.getGrid(3, 4).setColor(BLACK)
        self.getGrid(4, 3).setColor(BLACK)
        self.getGrid(4, 4).setColor(WHITE)
        self.__count = 4

    def startup_2(self):
        """放置中间四个初始棋子."""
        self.clean()
        self.getGrid(3, 3).setColor(BLACK)
        self.getGrid(3, 4).setColor(WHITE)
        self.getGrid(4, 3).setColor(WHITE)
        self.getGrid(4, 4).setColor(BLACK)
        self.__count = 4

    def getGrid(self, x: int, y: int) -> Grid:
        return self.__map[x][y]

    @staticmethod
    def isOnBoard(x: int, y: int) -> bool:
        return 0 <= x < 8 and 0 <= y < 8

    def gameMove(self, move: MoveResult) -> None:
        """
        将`MoveResult`对象的`color`下在`x`行`y`列位置，输入的`x`, `y`范围[0, 8).
        如果`move`无效，将不会对棋盘产生任何变化.
        """
        if not move:
            return
        x = move.x
        y = move.y
        color = move.color
        self.getGrid(x, y).setColor(color)
        is_flipped = False
        for dx, dy in ReversiMap.ReversiMapDirections:
            xx = x + dx
            yy = y + dy
            while self.isOnBoard(xx, yy):
                if self.getGrid(xx, yy).get() == color:
                    if xx != x + dx or yy != y + dy:
                        self._flip(x, y, xx, yy)
                        is_flipped = True
                    break
                elif self.getGrid(xx, yy).get() == EMPTY:
                    break
                else:
                    xx += dx
                    yy += dy
        if is_flipped:
            self.__count += 1
            return
        # BAD MOVE
        self.__map[x][y] = Grid()  # clean the grid
        # raise RuntimeError("没有可翻转的子，无法下在这个位置")
        return 2

    def _flip(self, x: int, y: int, xx: int, yy: int) -> None:
        """
        将(x, y)至(xx, yy)之间的（不包括两端）的子翻转.
        """
        if x != xx and y != yy and abs(x-xx) != abs(y-yy):
            raise RuntimeError("无法翻转不是横行、竖行、对角线三者之一的线上的子")
        dx = 1 if xx > x else (0 if xx == x else -1)
        dy = 1 if yy > y else (0 if yy == y else -1)
        while True:
            x += dx
            y += dy
            if x == xx and y == yy:
                break
            self.getGrid(x, y).reverse()

    def __iter__(self):
        """
        允许用`for`循环遍历棋盘.
        """
        for row in self.__map:
            yield row

    def __eq__(self, __o: "ReversiMap") -> bool:
        """
        比较两个棋盘是否相等.
        """
        if not isinstance(__o, ReversiMap):
            return False
        for x in range(8):
            for y in range(8):
                if self.getGrid(x, y).get() != __o.getGrid(x, y).get():
                    return False

        return True

    def isValidMove(self, move: MoveResult) -> None:
        """
        尝试是否能将`MoveResult`对象的`color`下在`x`行`y`列位置，输入的`x`, `y`范围[0, 8).
        不会对棋盘产生任何变化.
        """
        if not move:
            return False

        x = move.x
        y = move.y
        color = move.color

        if self.getGrid(x, y).get() != EMPTY:
            return False

        for dx, dy in ReversiMap.ReversiMapDirections:
            xx = x + dx
            yy = y + dy
            while self.isOnBoard(xx, yy):
                if self.getGrid(xx, yy).get() == color:
                    if xx != x + dx or yy != y + dy:
                        return True
                    break
                elif self.getGrid(xx, yy).get() == EMPTY:
                    break
                else:
                    xx += dx
                    yy += dy

        return False

    def getAllValidMoves(self, color: int) -> List[MoveResult]:
        """
        返回`color`所有有效的下棋位置.
        """
        moves = []
        for x in range(8):
            for y in range(8):
                move = MoveResult(x, y, color)
                if self.isValidMove(move):
                    moves.append(move)
        return moves
