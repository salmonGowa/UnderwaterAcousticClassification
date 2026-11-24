"""Save the public ShipsEar metadata listing as a CSV (metadata only, no audio).

Source: https://underwaternoise.atlanttic.uvigo.es/indexDB.php
Usage:  python data/scrape_shipsear_listing.py data/meta/shipsear_listing.csv
"""

import csv
import html
import re
import sys
import urllib.request

BASE = "https://underwaternoise.atlanttic.uvigo.es/indexDB.php"
N_ROWS = 90          # total stated in the listing's own pagination links
ROWS_PER_PAGE = 20

# Column order of the HTML table, as shown on the listing page.
COLUMNS = ["id", "name", "type", "pic", "location", "date", "h_g_d", "video",
           "channel_depth_m", "wind", "ais", "distance", "conditions",
           "duration", "notes"]


def cell_text(cell_html):
    text = re.sub(r"<[^>]+>", " ", cell_html)
    return " ".join(html.unescape(text).split())


def parse_page(page_html):
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", page_html, re.S):
        cells_html = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
        cells = [cell_text(c) for c in cells_html]
        if len(cells) != len(COLUMNS) or not cells[0].isdigit():
            continue
        row = dict(zip(COLUMNS, cells))
        row["location"] = row["location"].replace("Click for zoom", "").replace("click for zoom", "").strip()
        mmsi = re.search(r"mmsi=(\d+)", tr, re.I)
        row["mmsi"] = mmsi.group(1) if mmsi else ""
        minutes, seconds = row["duration"].split(":")
        row["duration_s"] = int(minutes) * 60 + int(seconds)
        rows.append(row)
    return rows


def main(out_path):
    rows = []
    n_pages = (N_ROWS + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE
    for page in range(n_pages):
        url = f"{BASE}?pageNum_leerMedidas={page}&totalRows_leerMedidas={N_ROWS}"
        with urllib.request.urlopen(url, timeout=60) as response:
            rows += parse_page(response.read().decode("utf-8", errors="replace"))
    if len(rows) != N_ROWS:
        raise RuntimeError(f"expected {N_ROWS} rows, parsed {len(rows)}")
    fields = ["id", "name", "type", "mmsi", "location", "date", "duration_s",
              "distance", "channel_depth_m", "h_g_d", "conditions", "notes"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: int(r["id"])))
    print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
