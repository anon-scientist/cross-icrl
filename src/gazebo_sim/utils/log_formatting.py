from typing import Iterable

def format_floating_arr(array: Iterable[float], depth: int = 2) -> str:
    # depth = number of digits after the decimal point
    return ", ".join(f"{x:.{depth}f}" for x in array)