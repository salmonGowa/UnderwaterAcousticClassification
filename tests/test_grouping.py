from src.data.grouping import groups_from_candidates


def test_ambiguous_candidates_are_merged_transitively():
    groups = groups_from_candidates({
        "r1": ["SHIP A"],
        "r2": ["SHIP B", "SHIP C"],   # ambiguous between B and C
        "r3": ["SHIP C"],
        "r4": ["SHIP D"],
    })
    assert groups["r2"] == groups["r3"]
    assert groups["r1"] != groups["r2"] != groups["r4"]


def test_group_names_do_not_depend_on_order():
    a = groups_from_candidates({"r1": ["Y", "X"], "r2": ["X"]})
    b = groups_from_candidates({"r2": ["X"], "r1": ["X", "Y"]})
    assert a == b == {"r1": "X", "r2": "X"}
