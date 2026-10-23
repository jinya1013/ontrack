"""Shortest exact-rejoin paths over a recorded trajectory."""

from __future__ import annotations


def shortest_suffix_first_jumps(max_reach: list[int]) -> tuple[list[int], list[int]]:
    """Find the shortest-to-terminal distance and optimal first jump for every state.

    An edge from ``t`` may end at any state in ``[t + 1, max_reach[t]]``.
    Backward dynamic programming therefore needs a range minimum over already
    solved suffix states. A segment tree makes all T range queries O(T log T).
    Ties prefer the farther jump, yielding the most contracted first action
    among equally short suffix paths.
    """
    terminal = len(max_reach)
    size = 1
    while size < terminal + 1:
        size *= 2
    infinity = terminal + 1
    tree = [(infinity, 0)] * (2 * size)

    def update(index: int, value: tuple[int, int]) -> None:
        position = size + index
        tree[position] = value
        while position > 1:
            position //= 2
            tree[position] = min(tree[2 * position], tree[2 * position + 1])

    def query(left: int, right: int) -> tuple[int, int]:
        result = (infinity, 0)
        left += size
        right += size
        while left < right:
            if left % 2:
                result = min(result, tree[left])
                left += 1
            if right % 2:
                right -= 1
                result = min(result, tree[right])
            left //= 2
            right //= 2
        return result

    distances = [infinity] * (terminal + 1)
    first_jumps = [-1] * terminal
    distances[terminal] = 0
    update(terminal, (0, -terminal))
    for source in range(terminal - 1, -1, -1):
        reach = int(max_reach[source])
        if not source + 1 <= reach <= terminal:
            raise ValueError(
                f"invalid exact-rejoin reach r[{source}]={reach} for T={terminal}"
            )
        suffix_distance, negative_target = query(source + 1, reach + 1)
        if suffix_distance == infinity:
            raise ValueError(f"state {source} cannot reach the terminal state")
        distances[source] = suffix_distance + 1
        first_jumps[source] = -negative_target
        update(source, (distances[source], -source))
    return distances, first_jumps
