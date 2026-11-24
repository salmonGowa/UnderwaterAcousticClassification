"""Vessel grouping on the real ShipsEar listing (data/meta/shipsear_listing.csv)."""

import pandas as pd
import pytest

from src.data.grouping import groups_from_candidates
from src.data.shipsear import LISTING_CSV, name_key, normalise_type, vessel_key
from src.evaluation import splits


@pytest.fixture(scope="module")
def listing():
    table = pd.read_csv(LISTING_CSV)
    table["key"] = table.apply(vessel_key, axis=1)
    table["label"] = table["type"].map(normalise_type)
    return table


@pytest.mark.parametrize("name, key", [
    ("Mar de Cangas (arrives, interference", "mardecangas"),
    ("Autopride(reverse)", "autopride"),
    ('Motorboat "Duda" (arrives)', "duda"),
    ('"Duda" (going out)', "duda"),
    ("Moanha tests sample 3", "adricristuy"),
    ("Mussel boat3 (interf)", "musselboat3"),
])
def test_name_key(name, key):
    assert name_key(name) == key


def test_same_named_ship_shares_one_group(listing):
    mar_de_onza = listing[listing["name"].str.startswith("Mar de Onza")]
    assert len(mar_de_onza) >= 5 and mar_de_onza["key"].nunique() == 1


def test_passenger_recordings_come_from_few_vessels(listing):
    # 30 recordings but only 7 ships: splitting by recording would leak.
    passengers = listing[listing["label"] == "Passengers"]
    assert len(passengers) == 30 and passengers["key"].nunique() == 7


def test_ambient_recordings_are_separate_groups(listing):
    ambient = listing[listing["label"] == "Natural ambient noise"]
    assert ambient["key"].nunique() == len(ambient) == 12


def test_no_group_mixes_labels(listing):
    assert (listing.groupby("key")["label"].nunique() == 1).all()


def test_grouped_split_on_real_listing_has_no_leak(listing):
    recs = listing.assign(recording_id=listing["id"].astype(str), target=listing["label"])
    groups = groups_from_candidates({r: [k] for r, k in zip(recs["recording_id"], recs["key"])})
    recs["group_id"] = recs["recording_id"].map(groups)
    binary = recs.assign(target=(recs["target"] == "Natural ambient noise").map({True: "ambient", False: "vessel"}))
    dev, test = splits.holdout_split(binary, 0.2, seed=0)
    splits.assert_no_group_overlap(binary, dev, test)
