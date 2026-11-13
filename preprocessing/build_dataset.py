"""Build one ML dataset (NPZ) from raw COMSOL eigenmode exports.

Every disorder realization is one folder holding the field along the inner line
cut (one complex column per candidate mode) and out.txt with the complex
eigenfrequency and Q factor of every mode. For each candidate mode:

  1. read the complex field E(z) on the non-uniform COMSOL mesh;
  2. take the upper envelope of |E|, Re E and Im E: straight lines between
     local maxima, no smoothing (chain: of |E| and the signed Re/Im, maxima at
     least 0.7 of the mean |E| peak spacing apart, one spacing per mode shared
     by all three; gst: of |E|, |Re|, |Im|, maxima by prominence);
  3. place the envelope on the common uniform grid of L = 1883 points;
  4. divide it by the peak of the |E| envelope (the divisor is saved as
     norm_max) and clamp the Re/Im envelopes at zero;
  5. attach meta = (sigma in %, Q, Re f, Im f, mode index, iteration, label).

The two structures of the paper differ only in steps 2-3 (envelope_chain and
envelope_gst):
  chain - 50 ceramic disks; the line cut starts at z = 50 mm, so the first 50
          grid points are zeros.
  gst   - Ge2Sb2Te5 bar metasurface; the structure is stretched over the last
          1833 points after 50 zeros, matching the chain layout.

The index CSV lists the realizations of the dataset in output order, with
columns folder, sigma_pct, iter, target_mode. target_mode is the 0-based index
of the target mode, -1 if the realization has no identifiable target, and
empty if the realization was not labeled (label NaN).

Usage:
  python preprocessing/build_dataset.py chain RAW_DIR labels/disk_chain.csv disk_chain.npz
  python preprocessing/build_dataset.py gst RAW_DIR labels/gst_crystalline_lossless.csv gst_crystalline_lossless.npz
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

L = 1883  # grid points per mode, both structures

# chain: uniform step in mm (the mesh step of the ordered chain); grid starts at z = 0
CHAIN_DX_MM = 1000 * 0.0010092927207021
CHAIN_GRID_MM = np.arange(L) * CHAIN_DX_MM

# gst: GST_LEFT zero points, then the structure stretched over GST_FILL points
GST_LEFT = 50
GST_FILL = L - GST_LEFT

CUT_FILE = {"chain": "cut1.txt", "gst": "cutline_inside.txt"}
COLUMNS = ["sigma", "quality", "freq_re", "freq_im", "meta_id", "iter", "target"]

# a real or complex number as COMSOL writes it: 2.2E9+9.5E5i, 1157.05
NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?([+-](\d+\.?\d*|\.\d+)([eE][+-]?\d+)?)?i?$")


def data_lines(text: str) -> list[list[str]]:
    """Whitespace-split rows of a COMSOL export, without the '%' header lines."""
    return [line.split() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("%")]


def read_cut(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Line-cut export -> x [M] (sorted) and complex field [M, K], one column per mode."""
    rows = data_lines(text)
    n_coords = 2 if "i" not in rows[0][1] else 1  # chain files carry (R, Z), GST files a single x
    coords = np.array([[float(t) for t in row[:n_coords]] for row in rows])
    field = np.array([[complex(t.replace("i", "j")) for t in row[n_coords:]] for row in rows])
    x = coords[:, np.argmax(coords.std(axis=0))]  # the coordinate that varies along the cut
    order = np.argsort(x)
    return x[order], field[order]


def read_out(text: str) -> list[tuple[float, float, float]]:
    """out.txt -> [(Re f, Im f, Q)] per mode; the file is a flat list of (frequency, Q) pairs."""
    tokens = [t for row in data_lines(text) for t in row if NUMBER.match(t)]
    freqs = [complex(t.replace("i", "j")) for t in tokens[0::2]]
    qs = [float(t) for t in tokens[1::2]]
    return [(f.real, f.imag, q) for f, q in zip(freqs, qs, strict=True)]  # odd count: corrupt file


def peak_spacing(e_abs: np.ndarray) -> int | None:
    """Minimum peak distance for the chain envelopes: 0.7 of the mean spacing of |E| maxima, in mesh points."""
    peaks = find_peaks(e_abs)[0]
    return max(1, int(np.mean(np.diff(peaks)) * 0.7)) if len(peaks) >= 2 else None


def envelope_chain(x: np.ndarray, y: np.ndarray, distance: int | None) -> np.ndarray:
    """Peaks on the mesh index, at least `distance` points apart; zero outside the cut."""
    peaks = find_peaks(y, distance=distance)[0]
    nodes = np.unique(np.concatenate(([0], peaks, [len(y) - 1])))
    env_on_mesh = np.interp(np.arange(len(y)), nodes, y[nodes])

    out = np.zeros(L)
    inside = (CHAIN_GRID_MM >= x[0]) & (CHAIN_GRID_MM <= x[-1])
    out[inside] = np.interp(CHAIN_GRID_MM[inside], x, env_on_mesh)
    return out


def envelope_gst(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Peaks by prominence (2% of the maximum) on the adaptive mesh, joined by straight lines in x."""
    peaks = find_peaks(y, prominence=0.02 * y.max())[0] if y.max() > 0 else np.array([], dtype=int)
    xp = np.concatenate(([x[0]], x[peaks], [x[-1]]))
    vp = np.concatenate(([y[0]], y[peaks], [y[-1]]))
    # COMSOL doubles mesh nodes at material interfaces; for a tied x keep the higher value
    order = np.lexsort((-vp, xp))
    xp, vp = xp[order], vp[order]
    first = np.concatenate(([True], xp[1:] > xp[:-1]))

    out = np.zeros(L)
    out[GST_LEFT:] = np.interp(np.linspace(x[0], x[-1], GST_FILL), xp[first], vp[first])
    return out


def build(kind: str, raw_dir: Path, index_csv: Path) -> dict[str, np.ndarray]:
    """Run steps 1-5 for every realization listed in index_csv; returns the NPZ arrays."""
    with open(index_csv, newline="") as f:
        index = list(csv.DictReader(f))

    e_abs, e_re, e_im, meta, extents = [], [], [], [], []
    for n, row in enumerate(index):
        folder = Path(raw_dir) / row["folder"]
        x, field = read_cut((folder / CUT_FILE[kind]).read_text(encoding="utf-8", errors="replace"))
        modes = read_out((folder / "out.txt").read_text(encoding="utf-8", errors="replace"))
        extents.append(x[-1] - x[0])

        target = row["target_mode"].strip()
        for j in range(field.shape[1]):
            if j >= len(modes) or not np.all(np.isfinite(modes[j])):
                continue  # no frequency/Q for this column: not a usable mode
            f_re, f_im, q = modes[j]
            if kind == "chain":
                spacing = peak_spacing(np.abs(field[:, j]))  # one spacing per mode, shared by |E|, Re, Im
                e_abs.append(envelope_chain(x, np.abs(field[:, j]), spacing))
                e_re.append(envelope_chain(x, field[:, j].real, spacing))
                e_im.append(envelope_chain(x, field[:, j].imag, spacing))
            else:
                e_abs.append(envelope_gst(x, np.abs(field[:, j])))
                e_re.append(envelope_gst(x, np.abs(field[:, j].real)))
                e_im.append(envelope_gst(x, np.abs(field[:, j].imag)))
            if f_im == 0.0 and f_re > 0.0 and q > 0.0:
                f_im = f_re / (2.0 * q)  # out.txt gave a real frequency (GST): use Q = Re f / (2 Im f)
            label = np.nan if target == "" else float(j == int(target))
            meta.append((float(row["sigma_pct"]), q, f_re, f_im, j, float(row["iter"]), label))
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(index)} realizations", flush=True)

    e_abs, e_re, e_im = np.array(e_abs), np.array(e_re), np.array(e_im)
    norm_max = e_abs.max(axis=1)
    if norm_max.min() <= 0.0:
        raise ValueError("a mode has an all-zero |E| envelope")

    if kind == "chain":
        dx = CHAIN_DX_MM * 1e-3
        x_grid = CHAIN_GRID_MM * 1e-3
    else:
        dx = float(np.median(extents)) / (GST_FILL - 1)
        x_grid = (np.arange(L) - GST_LEFT) * dx

    return {
        "E_abs": (e_abs / norm_max[:, None]).astype(np.float32),
        "E_Re": (np.maximum(e_re, 0.0) / norm_max[:, None]).astype(np.float32),
        "E_Im": (np.maximum(e_im, 0.0) / norm_max[:, None]).astype(np.float32),
        "meta": np.array(meta, dtype=np.float64),
        "columns_meta": np.array(COLUMNS),
        "x_grid": x_grid,
        "dx": np.float64(dx),
        "norm_max": norm_max,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("kind", choices=["chain", "gst"])
    p.add_argument("raw_dir", type=Path, help="root of the raw COMSOL exports")
    p.add_argument("index", type=Path, help="index CSV: folder, sigma_pct, iter, target_mode")
    p.add_argument("out", type=Path, help="output .npz")
    args = p.parse_args()

    data = build(args.kind, args.raw_dir, args.index)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **data)
    print(f"saved {args.out}: {data['E_abs'].shape[0]} modes x {L} points")


if __name__ == "__main__":
    main()
