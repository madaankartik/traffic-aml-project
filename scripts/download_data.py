#!/usr/bin/env python3
"""Download METR-LA speed data and the DCRNN sensor graph into data/.

The files are the ones released with DCRNN (Li et al., ICLR 2018).  Because the
original Google-Drive links are not scriptable, we fetch public GitHub mirrors
(see trafficml.config.DOWNLOAD_URLS) and verify the shapes after download.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trafficml import config as C   # noqa: E402

DEST = {
    "metr-la.h5": C.RAW_DIR,
    "graph_sensor_locations.csv": C.GRAPH_DIR,
    "distances_la_2012.csv": C.GRAPH_DIR,
    "graph_sensor_ids.txt": C.GRAPH_DIR,
    "adj_mx.pkl": C.GRAPH_DIR,
}


def main() -> None:
    for name, url in C.DOWNLOAD_URLS.items():
        target = DEST[name] / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            print(f"[skip] {target} exists")
            continue
        print(f"[get ] {url}")
        urllib.request.urlretrieve(url, target)
        print(f"       -> {target} ({target.stat().st_size / 1e6:.1f} MB)")

    from trafficml import data
    speed, raw, graph = data.load_all()
    assert speed.shape == (34272, 207), speed.shape
    assert graph.adjacency.shape == (207, 207)
    print(f"OK: speed {speed.shape}, {speed.index[0]} -> {speed.index[-1]}, graph edges {(graph.adjacency > 0).sum()}")


if __name__ == "__main__":
    main()
