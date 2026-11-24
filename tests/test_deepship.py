from src.data.deepship import candidate_ships

META = [
    {"file_id": "5", "ship_name": "ALPHA", "meta_duration_s": 300},
    {"file_id": "5", "ship_name": "BRAVO", "meta_duration_s": 40},   # id restarts in a later block
    {"file_id": "6", "ship_name": "CHARLIE", "meta_duration_s": 250},
    {"file_id": "7", "ship_name": "DELTA", "meta_duration_s": 100},
    {"file_id": "7", "ship_name": "ECHO", "meta_duration_s": 120},
]


def test_unique_id():
    assert candidate_ships("6", 200, META) == (["CHARLIE"], "unique match")


def test_duration_resolves_restarted_id():
    # A 200 s file cannot come from the 40 s recording.
    assert candidate_ships("5", 200, META) == (["ALPHA"], "resolved by duration")


def test_still_ambiguous_ids_return_every_candidate():
    names, note = candidate_ships("7", 50, META)
    assert names == ["DELTA", "ECHO"] and note.startswith("ambiguous")


def test_missing_id():
    assert candidate_ships("99", 10, META) == ([], "no metafile row")
