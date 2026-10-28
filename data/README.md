# data/

METR-LA (Jagadish et al., 2014; released with DCRNN, Li et al., ICLR 2018): average vehicle speed from
207 loop detectors on Los Angeles County highways, every 5 minutes, 1 March – 27 June 2012, plus the
road-network sensor graph. `scripts/download_data.py` fetches the files below from public mirrors of the
DCRNN release and verifies the shapes.

## Files

| Path | What it is | Size | Committed |
|---|---|---|---|
| `raw/metr-la.h5` | speed table, 34,272 timestamps × 207 sensors (mph), HDF5 | 57 MB | no — downloaded |
| `sensor_graph/graph_sensor_ids.txt` | the 207 sensor IDs, in the column order of the speed table | 1 KB | yes |
| `sensor_graph/graph_sensor_locations.csv` | sensor_id, latitude, longitude | 6 KB | yes |
| `sensor_graph/distances_la_2012.csv` | from, to, cost: directed road-network distance in metres, 295,374 pairs | 6.4 MB | yes |
| `sensor_graph/adj_mx.pkl` | DCRNN adjacency: A_ij = exp(−(d_ij/σ)²), thresholded at 0.1 (1,515 directed edges, mean out-degree 7.3) | 680 KB | yes |

Sources: `raw/metr-la.h5` — `github.com/deepkashiwa20/DL-Traff-Graph` (METRLA/metr-la.h5);
`sensor_graph/*` — `github.com/liyaguang/DCRNN` (data/sensor_graph/). The adjacency construction is
re-implemented in `trafficml.data.build_adjacency` and matches the released matrix exactly.

SHA-256:

```
metr-la.h5                  64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05
adj_mx.pkl                  a35687c6e15aa228dc45027b0ed2a0ea0f4ec78f573deb992c595092d12f61b3
distances_la_2012.csv       a576a2a3e28dbb959be6da22688e24dd1b246b81264595e129147c256cd53de5
graph_sensor_ids.txt        3ba026caa2e6263ab0ea54b0fa1b125dbfa7216544cd05313b555e826292b990
graph_sensor_locations.csv  eb8ea96e07358b45d0e4ba3b89c2673fa20c54af50150249e627389e749ade6f
```

## Data-quality audit (full numbers in `results/eda_summary.json`, per sensor in `results/eda_sensor_table.csv`)

- 34,272 × 207 = 7,094,304 readings; no missing timestamps in the index; speeds 0–70 mph.
- **8.1 % of readings are exactly 0.0 mph.** A 5-minute highway average is never truly zero; these are
  detector/communication outages and are **masked as missing (NaN)**, never used as a feature value or a
  target, and excluded from every metric. No row or sensor is dropped.
- Outages come in 25,963 runs (median 2 steps, 90th percentile 71, longest 1,683 = 5.8 days); 22.5 % of
  missing steps sit in runs of a day or longer, and 6.3 % of timestamps have every sensor missing
  (network-wide outages). Only 1 sensor exceeds 20 % missing (20.1 %).
- Observed speeds: mean 58.5 mph, median 63.2 mph; 9.0 % below the 35 mph congestion threshold
  (Caltrans PeMS convention). The distribution is bimodal (free-flow vs congested).

## Split (chronological; `trafficml.features.chronological_split`, logged in `results/split_summary.csv`)

| Split | Share | Sample timestamps | From | To |
|---|---|---|---|---|
| train | 70 % | 23,967 | 2012-03-01 00:55 | 2012-05-23 06:05 |
| validation | 10 % | 3,403 | 2012-05-23 08:10 | 2012-06-04 03:40 |
| test | 20 % | 6,831 | 2012-06-04 05:45 | 2012-06-27 22:55 |

Each sample timestamp covers all 207 sensors. A one-hour gap separates the partitions so that no training
target (up to +60 min) overlaps a validation/test feature window. Sequences are never shuffled: random
k-fold is invalid for serially dependent data (Bergmeir & Benítez, 2012). Statistics that could leak the
future (the historical-average profile, Ridge standardisation) are fitted on training timestamps only.
