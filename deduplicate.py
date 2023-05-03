def rotate1(s: str):
    # (j, 7-i)
    if len(s) != 64:
        raise ValueError("无效输入")

    ans = ""
    for i in range(8):
        for j in range(8):
            ans += s[8 * j + 7 - i]
    return ans


def rotate2(s: str):
    # (7-i, 7-j)
    if len(s) != 64:
        raise ValueError("无效输入")

    ans = ""
    for i in range(8):
        for j in range(8):
            ans += s[8 * (7 - i) + 7 - j]
    return ans


def rotate3(s: str):
    # (7-j, i)
    if len(s) != 64:
        raise ValueError("无效输入")

    ans = ""
    for i in range(8):
        for j in range(8):
            ans += s[8*(7-j) + i]
    return ans


def symmetry(s: str):
    # (j, i)
    if len(s) != 64:
        raise ValueError("无效输入")
    ans = ""
    for i in range(8):
        for j in range(8):
            ans += s[8 * j + i]
    return ans


def symrot1(s: str):
    # (7-i, j)
    if len(s) != 64:
        raise ValueError("无效输入")
    ans = ""
    for i in range(8):
        for j in range(8):
            ans += s[8 * (7 - i) + j]
    return ans


def symrot2(s: str):
    # (7-j, 7-i)
    if len(s) != 64:
        raise ValueError("无效输入")
    ans = ""
    for i in range(8):
        for j in range(8):
            ans += s[8 * (7 - j) + 7 - i]
    return ans


def symrot3(s: str):
    # (i, 7-j)
    if len(s) != 64:
        raise ValueError("无效输入")
    ans = ""
    for i in range(8):
        for j in range(8):
            ans += s[8 * i + 7 - j]
    return ans
