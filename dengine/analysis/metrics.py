def recall(TP: int, FP: int, FN: int, TN: int) -> float:
    if TP == 0:
        return 0
    return (TP) / (TP + FN)


def precision(TP: int, FP: int, FN: int, TN: int) -> float:
    if TP == 0:
        return 0
    return (TP) / (TP + FP)


def f1(TP: int, FP: int, FN: int, TN: int) -> float:
    if TP == 0:
        return 0
    return (
        (2 * TP) /
        (2 * TP + FP + FN)
    )
