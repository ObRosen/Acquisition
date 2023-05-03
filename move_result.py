class MoveResult(object):
    __slots__ = ("__x", "__y", "__color", "__valid")

    def __init__(self, x: int, y: int, color: int, valid: bool = True) -> None:
        self.__x = x
        self.__y = y
        self.__color = color
        self.__valid = valid

    @property
    def x(self) -> int:
        return self.__x

    @property
    def y(self) -> int:
        return self.__y

    @property
    def color(self) -> int:
        return self.__color

    def isValid(self) -> bool:
        return self.__valid

    def __bool__(self) -> bool:
        return self.__valid
