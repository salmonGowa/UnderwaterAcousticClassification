# Data

No audio is stored in this repository. Neither ShipsEar nor DeepShip states a licence, so we
redistribute nothing; we provide download scripts, checksums, and the public ShipsEar metadata listing.

Set `UAC_DATA_ROOT` to where the data should live (on Colab: a Google Drive folder). Layout:

```
$UAC_DATA_ROOT/raw/deepship/<Cargo|Passengership|Tanker|Tug>/<id>.wav + <class>-metafile
$UAC_DATA_ROOT/raw/shipsear/<id>__<date>_<name>.wav
$UAC_DATA_ROOT/manifests/   built by src.run
$UAC_DATA_ROOT/cache/       built by src.run (features; never recomputed during training)
```

## Openly available now

```
python data/download.py deepship_github  --data-root $UAC_DATA_ROOT   # 63 files, ~0.7 GB, truncated at ~200 s
python data/download.py shipsear_samples --data-root $UAC_DATA_ROOT   # 4 sample files, 84 MB zip
```

## Full datasets (email request)

| Dataset | Request to | Source |
|---|---|---|
| ShipsEar | dsantos@gts.uvigo.es | https://underwaternoise.atlanttic.uvigo.es/ |
| DeepShip | mirfan@mail.nwpu.edu.cn | https://github.com/irfankamboh/DeepShip |

When the files arrive, place them in the layout above, then record their checksums once:

```
python data/download.py checksums deepship --data-root $UAC_DATA_ROOT
python data/download.py checksums shipsear --data-root $UAC_DATA_ROOT
```

This writes `data/checksums/<dataset>_full.sha256` (commit it). Later runs verify against it.
If the full DeepShip folder replaces the GitHub subset, delete `$UAC_DATA_ROOT/manifests/deepship.csv`
(or pass `--rebuild-manifests`) so the manifest is rebuilt.

## Metadata files in this folder

- `meta/shipsear_listing.csv` — the 90-row public ShipsEar listing (id, name, type, MMSI, location, date,
  duration, ...), scraped by `scrape_shipsear_listing.py` on 2026-10-27. Used for labels and vessel groups.
- `checksums/*.sha256` — SHA-256 of every downloaded file.

## Citations

- D. Santos-Domínguez et al., "ShipsEar: An underwater vessel noise database," *Applied Acoustics* 113 (2016) 64–69. doi:10.1016/j.apacoust.2016.06.008
- M. Irfan et al., "DeepShip: An underwater acoustic benchmark dataset and a separable convolution based autoencoder for classification," *Expert Systems with Applications* (2021). doi:10.1016/j.eswa.2021.115270
