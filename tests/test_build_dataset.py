import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "preprocessing"))
import build_dataset as bd

COMSOL_HEADER = "% Model: x.mph\n% Version: COMSOL 6.0.0.318\n% Date: Sep 5 2023, 23:56\n"


def write_cut(path: Path, coords: list[np.ndarray], fields: np.ndarray) -> None:
    """COMSOL line-cut export: coordinate columns, then one complex column per mode."""
    lines = [COMSOL_HEADER + "% R  Z  ewfd2.Ephi @ lambda=1 ..."]
    for i in range(fields.shape[0]):
        row = [repr(float(c[i])) for c in coords]
        row += [f"{float(v.real)!r}{float(v.imag):+.17g}i" for v in fields[i]]
        lines.append("   ".join(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_index(path: Path, rows: list[tuple]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["folder", "sigma_pct", "iter", "target_mode"])
        w.writerows(rows)


def standing_waves(z: np.ndarray, n_modes: int) -> np.ndarray:
    """Fast carrier under a slow sine envelope, a different envelope per mode."""
    span = z[-1] - z[0]
    cols = [np.sin(np.pi * (k + 1) * (z - z[0]) / span) * np.cos(2 * np.pi * z / 37.0) for k in range(n_modes)]
    return np.stack(cols, axis=1) * (1.0 + 0.3j)


def test_read_out_skips_comsol_date_line():
    text = COMSOL_HEADER + "% freq (Hz)  Q\n2.2E9+9.5E5i   1157.05   2.3E9+7.8E5i  1409.13\n"
    assert bd.read_out(text) == [(2.2e9, 9.5e5, 1157.05), (2.3e9, 7.8e5, 1409.13)]


def test_read_cut_picks_varying_coordinate_and_complex_columns(tmp_path):
    z = np.array([50.0, 52.5, 51.0, 60.0])
    field = np.array([[1 + 2j, 3 - 1j], [0.5j, 2], [-1, 1j], [2 - 2j, 0]])
    write_cut(tmp_path / "cut.txt", [np.full(4, 7.5), z], field)
    x, e = bd.read_cut((tmp_path / "cut.txt").read_text())
    assert np.array_equal(x, np.sort(z))
    assert np.array_equal(e, field[np.argsort(z)])


def test_envelope_chain_touches_interior_peaks():
    x = np.linspace(50.0, 1900.0, 4000)
    y = np.abs(np.sin(np.pi * (x - 50) / 1850)) * np.abs(np.cos(2 * np.pi * x / 37.0))
    env = bd.envelope_chain(x, y, bd.peak_spacing(y))
    inside = (bd.CHAIN_GRID_MM >= x[0]) & (bd.CHAIN_GRID_MM <= x[-1])
    assert env.shape == (bd.L,)
    assert np.all(env[~inside] == 0.0)
    assert np.all(env[inside] >= -1e-12)
    assert abs(env.max() - y.max()) < 1e-3


def test_build_chain_end_to_end(tmp_path):
    z = np.concatenate([np.linspace(50.0, 100.0, 300), np.linspace(100.5, 1900.0, 2500)])
    write_cut(tmp_path / "raw/sigma10/iter1/cut1.txt", [np.full(z.size, 7.5), z], standing_waves(z, 2))
    (tmp_path / "raw/sigma10/iter1/out.txt").write_text(COMSOL_HEADER + "2.2E9+9.5E5i 1157.0 2.3E9+7.8E5i 1409.0\n")
    write_index(tmp_path / "index.csv", [("sigma10/iter1", 1.0, 1, 1)])

    d = bd.build("chain", tmp_path / "raw", tmp_path / "index.csv")

    assert d["E_abs"].shape == (2, bd.L) and d["E_abs"].dtype == np.float32
    assert np.allclose(d["E_abs"].max(axis=1), 1.0)
    assert np.all(d["E_Re"] >= 0) and np.all(d["E_Im"] >= 0)
    assert np.all(d["E_abs"][:, :50] == 0.0)  # the line cut starts at z = 50 mm
    assert np.allclose(d["x_grid"][1] - d["x_grid"][0], d["dx"]) and abs(d["x_grid"][-1] - 1.8995) < 1e-3
    assert d["meta"][:, 0].tolist() == [1.0, 1.0]
    assert d["meta"][:, 1].tolist() == [1157.0, 1409.0]
    assert d["meta"][:, 4].tolist() == [0, 1] and d["meta"][:, 6].tolist() == [0, 1]


def test_build_gst_fills_freq_im_and_leaves_unlabeled_nan(tmp_path):
    x = np.linspace(0.0, 1.4e-5, 900)
    write_cut(tmp_path / "raw/NEW_sigma_10/iter_0/cutline_inside.txt", [x], standing_waves(x * 1e7, 3))
    (tmp_path / "raw/NEW_sigma_10/iter_0/out.txt").write_text(COMSOL_HEADER + "1E14 50\n2E14 100\n3E14 25\n")
    write_index(tmp_path / "index.csv", [("NEW_sigma_10/iter_0", 1.0, 0, "")])

    d = bd.build("gst", tmp_path / "raw", tmp_path / "index.csv")

    assert d["E_abs"].shape == (3, bd.L)
    assert np.all(d["E_abs"][:, : bd.GST_LEFT] == 0.0) and d["x_grid"][bd.GST_LEFT] == 0.0
    assert np.allclose(d["x_grid"][-1], 1.4e-5)
    assert np.allclose(d["meta"][:, 3], [1e12, 1e12, 6e12])  # f_im = f_re / (2 Q)
    assert np.all(np.isnan(d["meta"][:, 6]))


def test_target_minus_one_means_labeled_without_target(tmp_path):
    x = np.linspace(0.0, 1.0, 500)
    write_cut(tmp_path / "raw/s/i/cutline_inside.txt", [x], standing_waves(x * 300, 2))
    (tmp_path / "raw/s/i/out.txt").write_text("1E14 50\n2E14 100\n")
    write_index(tmp_path / "index.csv", [("s/i", 2.0, 0, -1)])

    d = bd.build("gst", tmp_path / "raw", tmp_path / "index.csv")

    assert d["meta"][:, 6].tolist() == [0.0, 0.0]


def test_build_chain_reproduces_published_rows():
    """A real realization (sigma = 6 %, iter 1, raw COMSOL export) rebuilds to the published disk_chain.npz rows."""
    data = Path(__file__).resolve().parent / "data"
    d = bd.build("chain", data / "raw", data / "chain_sigma60_iter1.csv")
    ref = np.load(data / "chain_sigma60_iter1_expected.npz")

    # |E| of a complex number can differ by one float64 ulp between numpy builds, i.e. one float32 step after scaling
    for key in ["E_abs", "E_Re", "E_Im"]:
        np.testing.assert_allclose(d[key], ref[key], rtol=0, atol=2e-7, err_msg=key)
    np.testing.assert_array_equal(d["meta"], ref["meta"])
    np.testing.assert_allclose(d["norm_max"], ref["norm_max"], rtol=1e-12)
