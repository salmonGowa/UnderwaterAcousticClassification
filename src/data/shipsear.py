"""Build the ShipsEar manifest from the wav files plus the scraped public listing.

The site says: "The first number in the filename of the wave files matches their
'id' in the database", e.g. "6__10_07_13_marDeCangas_Entra.wav" -> id 6.

Vessel key: the AIS MMSI when the listing has one, otherwise the ship name with
suffixes such as "(arrives)" removed. Generic names ("Motorboat", "Sailboat") then
share a group, which may merge different boats; that is the safe direction.
"""

import re
from pathlib import Path

import pandas as pd
import soundfile as sf

from src.data.grouping import groups_from_candidates

LISTING_CSV = Path(__file__).resolve().parents[2] / "data" / "meta" / "shipsear_listing.csv"

# Same vessel under different names, stated in the listing's own notes column.
NAME_OVERRIDES = {
    "moanhatestssample1": "adricristuy",  # notes: "Dredger Adricristuy recording during Moanha ..."
    "moanhatestssample2": "adricristuy",
    "moanhatestssample3": "adricristuy",
    "moanhatestssample4": "adricristuy",
    "duda": "duda",
    "motorboatduda": "duda",              # notes: "Motorboat Duda arrives."
}


def name_key(name):
    """'Mar de Cangas (arrives, interference' -> 'mardecangas'."""
    name = re.sub(r"\(.*$", "", name)            # drop everything from the first "("
    key = re.sub(r"[^a-z0-9]", "", name.lower())  # drop spaces, quotes, punctuation
    return NAME_OVERRIDES.get(key, key)


def normalise_type(raw_type):
    """The listing has both 'Fishboat' and 'fishboat'."""
    raw_type = raw_type.strip()
    return "Fishboat" if raw_type.lower() == "fishboat" else raw_type


def vessel_key(row):
    if str(row["type"]) == "Natural ambient noise":
        return f"shipsear:ambient:{row['id']}"   # no vessel: one group per recording
    if pd.notna(row["mmsi"]) and str(row["mmsi"]).strip():
        return f"shipsear:mmsi:{int(row['mmsi'])}"
    return f"shipsear:name:{name_key(row['name'])}"


def build_manifest(data_root, listing_csv=LISTING_CSV):
    raw = Path(data_root) / "raw" / "shipsear"
    listing = pd.read_csv(listing_csv).set_index("id")
    records, candidates = [], {}
    for wav in sorted(raw.rglob("*.wav")):
        if wav.name.startswith("._"):
            continue  # macOS resource-fork files shipped inside the zip
        file_id = int(wav.name.split("_")[0])
        row = listing.loc[file_id].copy()
        row["id"] = file_id
        recording_id = f"shipsear/{file_id}"
        key = vessel_key(row)
        candidates[recording_id] = [key]
        records.append({
            "recording_id": recording_id,
            "dataset": "shipsear",
            "path": str(wav.relative_to(data_root)),
            "label": normalise_type(row["type"]),
            "group_note": key.split(":")[1],      # "mmsi", "name" or "ambient"
            "duration_s": round(sf.info(str(wav)).duration, 3),
        })
    if not records:
        raise FileNotFoundError(f"no ShipsEar wav files under {raw}")
    groups = groups_from_candidates(candidates)
    manifest = pd.DataFrame(records)
    manifest["group_id"] = manifest["recording_id"].map(groups)
    return manifest
