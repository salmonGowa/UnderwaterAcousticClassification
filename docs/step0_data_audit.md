# Step 0 — Data audit (2026-10-27)

Every figure below was read from an official page or computed from an official
file (method in brackets). Anything not confirmed is marked **unverified**.

## Summary table

| | ShipsEar | DeepShip | Watkins (WMMSD) |
|---|---|---|---|
| Official source | [site](https://underwaternoise.atlanttic.uvigo.es/), [listing](https://underwaternoise.atlanttic.uvigo.es/indexDB.php) | [GitHub](https://github.com/irfankamboh/DeepShip), [README](https://github.com/irfankamboh/DeepShip/blob/main/README.txt) | [site](https://cis.whoi.edu/science/B/whalesounds/index.cfm) |
| Access | Browsable metadata; 4-file `samples.zip` (84 MB) is open. **Full audio by email request** to dsantos@gts.uvigo.es | **Partial** on GitHub (63 WAVs, 0.74 GB [GitHub tree API]); **rest by email request** to mirfan@mail.nwpu.edu.cn | **Unverified**: site "Under Maintenance" on 2026-10-27; library record behind a CAPTCHA |
| Total size | Not stated | Not stated; ~21.7 GB estimated (47.07 h × 32 kHz × 4 B) | Unverified |
| Format | WAV, mono, 24-bit, **52,734 Hz** [header of all 4 sample files] | WAV, mono, 32-bit float, **32,000 Hz** [header of 1 Tug file] | Unverified |
| Size of set | 90 recordings, 188.3 min [parsed listing] | 611 recordings, 47.07 h [metafiles]; README: "47 hours and 04 minutes … 265 different ships" | Unverified |
| Vessel / source IDs | `name` field (ship name, with suffixes such as "(arrives)"); MMSI links for some; date, location, distance, depth, wind | Per-class metafile: id, AIS-like type code, ship name, date, time, duration (s), unknown pair. No header; meaning of columns **inferred** | Unverified |
| Licence | **Not stated** ("©Uvigo-Sonitum 2016" only) | **Not stated**; no LICENSE file in the repo tree | Unverified |
| Citation | Santos-Domínguez et al., "ShipsEar: An underwater vessel noise database", *Applied Acoustics* 113 (2016) 64–69, doi:10.1016/j.apacoust.2016.06.008 | Irfan et al. (2021), "DeepShip: …", *Expert Systems with Applications*, doi:[10.1016/j.eswa.2021.115270](https://doi.org/10.1016/j.eswa.2021.115270) | Unverified |

ShipsEar paper (open copy at https://hdl.handle.net/11093/10832): server refused the connection, so its tables and its 5-class A–E grouping are **unverified**.

## Per-class counts

**ShipsEar** (parsed from all 90 rows of the official listing; "vessels" = distinct `name` strings, an upper bound because one ship can appear under several names):

| type | recordings | minutes | names |
|---|---|---|---|
| Passengers | 30 | 71.2 | 28 |
| RORO | 5 | 25.2 | 5 |
| Natural ambient noise | 12 | 19.0 | 12 |
| Motorboat | 13 | 16.9 | 12 |
| Ocean liner | 7 | 15.7 | 7 |
| Mussel boat | 5 | 12.2 | 5 |
| Sailboat | 4 | 6.8 | 3 |
| Fishboat (+1 "fishboat") | 3 + 1 | 6.2 + 2.4 | 4 |
| Dredger | 5 | 4.4 | 5 |
| Tugboat | 2 | 3.4 | 2 |
| Trawler | 1 | 2.7 | 1 |
| Pilot ship | 2 | 2.3 | 1 |

Recorded 2012-10-04 to 2014-03-06 around the port and ría of Vigo, Spain.

**DeepShip** (parsed from the four official metafiles; "names" = distinct ship names, some truncated, e.g. "QUEEN OF"):

| class | recordings | hours | names | GitHub WAVs |
|---|---|---|---|---|
| Cargo | 110 | 10.58 | 62 | 12 |
| Passengership | 189 | 12.18 | 44 | 20 |
| Tanker | 243 | 12.98 | 130 | 28 |
| Tug | 69 | 11.33 | 16 | 3 |
| **Total** | 611 | 47.07 | 252 (README: 265) | 63 |

Recorded 2016-05-02 to 2018-10-04 at the "strait of Georgia delta node" (README).
Inference: field 2 matches AIS ship-type codes (52 tug, 6x passenger, 7x cargo, 8x tanker), and field 6 is duration in seconds (sums to 47.07 h, matching the README).
Risk: the id numbering restarts inside the Cargo and Tanker metafiles, so the mapping from WAV filename to ship must be checked once the full data is available.
