"""Assign each recording a group ID so that one vessel never sits on both sides of a split.

A recording may have several candidate vessels when the metadata is ambiguous
(e.g. a DeepShip file id that matches rows for different ships). We put all
candidates into one group with a union-find. Merging is the safe direction:
it can only make the split stricter, never leak a vessel.
"""


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]  # path halving
            item = self.parent[item]
        return item

    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            # Keep the alphabetically smaller root so group names are deterministic.
            low, high = sorted([root_a, root_b])
            self.parent[high] = low


def groups_from_candidates(candidates_by_recording):
    """Map recording_id -> group_id, given recording_id -> list of candidate vessel keys."""
    uf = UnionFind()
    for recording_id, candidates in candidates_by_recording.items():
        if not candidates:
            raise ValueError(f"recording {recording_id} has no candidate vessel key")
        for key in candidates:
            uf.find(key)
        for key in candidates[1:]:
            uf.union(candidates[0], key)
    return {rec: uf.find(cands[0]) for rec, cands in candidates_by_recording.items()}
