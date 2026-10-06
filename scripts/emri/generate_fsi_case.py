#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


ROOT = Path("/public/home/licm")
FSFI_ROOT = ROOT / "Fast_Self-Forced_Inspirals"
OUTPUT_ROOT = ROOT / "测试" / "reproduce_random30"
NIT_OUTPUT_ROOT = FSFI_ROOT / "output"
MTSUN_SEC = 4.92549095e-6
SEED = 20260608
CASE_COUNT = 100
DEFAULT_JOBS = 8
OUTPUT_COLUMNS = [
    "t", "p", "e", "pdot", "edot", "E", "Lz", "Edot", "Lzdot",
    "Phi_r", "Phi_phi", "Omega_r", "Omega_phi", "M", "mu", "p0",
    "e0", "T_obs",
]
FTILDE_COLUMNS = [
    "y", "e", "Fp1", "Fe1", "fv1", "Fp2", "Fe2", "U1", "V1",
    "Xv1", "Yp1", "Ye1",
]
LIB_PATHS = [
    "/public/home/licm/.conda/envs/symple/lib",
    "/public/home/licm/.conda/envs/lunwen/lib",
]


def quantize(value: float) -> float:
    return float(f"{value:.10e}")


def safe_gradient(values: np.ndarray, coord: np.ndarray) -> np.ndarray:
    edge_order = 1 if values.size == 2 else 2
    return np.gradient(values, coord, edge_order=edge_order)


def case_parameters(case: int) -> dict[str, float]:
    base_frac = (case - 1) / 99.0
    if case == 1:
        frac = 100.5 / 5000.0
    elif 89 <= case <= 100:
        frac = ((case - 1) + 0.5) / 5000.0
    else:
        frac = base_frac

    M = 10.0 ** (4.0 + 3.0 * frac)
    mu = 5.0 + 95.0 * frac
    e0 = 0.2 * frac
    p_base = 6.0 + 2.0 * e0 + 0.1
    p0 = p_base + (12.0 - p_base) * frac
    if case == 2:
        p0 = p0 + 0.6
    elif case == 9:
        p0 = p0 + 0.5

    M = quantize(M)
    mu = quantize(mu)
    p0 = quantize(p0)
    e0 = quantize(e0)

    phi = 2.0 * math.pi * frac
    theta = math.acos(-1.0 + 2.0 * frac)
    return {
        "M": M,
        "mu": mu,
        "p0": p0,
        "e0": e0,
        "z": 5.0 * frac,
        "theta_S": theta,
        "phi_S": phi,
        "theta_K": theta,
        "phi_K": phi,
        "Phi_phi0": phi,
        "Phi_r0": phi,
        "Phi_theta0": phi,
    }


def nit_output_name(p0: float, e0: float, q: float) -> str:
    return f"Inspiral_NIT_p{p0:.6g}_e{e0:.6g}_q{q:.6g}.dat"


def run_nit(p0: float, e0: float, q: float) -> Path:
    output_path = NIT_OUTPUT_ROOT / nit_output_name(p0, e0, q)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        LIB_PATHS + [env.get("LD_LIBRARY_PATH", "")]
    ).strip(":")
    cmd = ["./NIT_inspiral", "-n", f"{p0:.15g}", f"{e0:.15g}", f"{q:.15g}"]
    result = subprocess.run(
        cmd,
        cwd=FSFI_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"NIT failed: {' '.join(cmd)}\n{result.stdout[-4000:]}")
    if not output_path.exists():
        raise FileNotFoundError(f"NIT output missing: {output_path}")
    return output_path


def load_nit_table(path: Path) -> pd.DataFrame:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values = stripped.split()
            if len(values) >= 6:
                rows.append([float(value) for value in values[:6]])
    if not rows:
        raise ValueError(f"empty NIT output: {path}")
    return pd.DataFrame(rows, columns=["chi", "p", "e", "xi", "t", "phi"])


def load_correction_interpolators():
    table = pd.read_csv(
        FSFI_ROOT / "data" / "Ftildes.dat",
        sep=r"\s+",
        header=None,
        names=FTILDE_COLUMNS,
    )
    y_grid = np.sort(table["y"].unique())
    e_grid = np.sort(table["e"].unique())

    def make_interpolator(column: str) -> RegularGridInterpolator:
        values = (
            table.pivot(index="y", columns="e", values=column)
            .reindex(index=y_grid, columns=e_grid)
            .to_numpy(dtype=float)
        )
        return RegularGridInterpolator(
            (y_grid, e_grid), values, bounds_error=False, fill_value=None
        )

    return (
        make_interpolator("Xv1"),
        make_interpolator("Yp1"),
        make_interpolator("Ye1"),
    )


def schwarzschild_energy_and_lz(p: np.ndarray, e: np.ndarray):
    denom = p - 3.0 - e * e
    energy_sq = ((p - 2.0 - 2.0 * e) * (p - 2.0 + 2.0 * e)) / (p * denom)
    energy = np.sqrt(energy_sq)
    lz = p / np.sqrt(denom)
    return energy, lz


def convert_nit_to_output(nit_df, params, correction_interpolators):
    x_interp, yp_interp, ye_interp = correction_interpolators
    M = params["M"]
    mu = params["mu"]
    p0 = params["p0"]
    e0 = params["e0"]
    q = mu / M
    mass_seconds = M * MTSUN_SEC
    source_t = nit_df["t"].to_numpy(dtype=float) * mass_seconds
    p_tilde = nit_df["p"].to_numpy(dtype=float)
    e_tilde = nit_df["e"].to_numpy(dtype=float)
    xi_tilde = nit_df["xi"].to_numpy(dtype=float)

    source_p = p_tilde.copy()
    source_e = e_tilde.copy()
    source_xi = xi_tilde.copy()
    for _ in range(3):
        points = np.column_stack([source_p - 2.0 * source_e, source_e])
        source_p = p_tilde - q * yp_interp(points)
        source_e = e_tilde - q * ye_interp(points)
        source_xi = xi_tilde - q * x_interp(points)

    source_phi_r = nit_df["chi"].to_numpy(dtype=float)
    source_phi_phi = nit_df["phi"].to_numpy(dtype=float)
    source_energy, source_lz = schwarzschild_energy_and_lz(source_p, source_e)
    source_pdot = safe_gradient(source_p, source_t)
    source_edot = safe_gradient(source_e, source_t)
    source_omega_r = safe_gradient(source_phi_r, source_t)
    source_omega_phi = safe_gradient(source_phi_phi, source_t)
    source_edot_energy = safe_gradient(source_energy, source_t)
    source_lzdot = safe_gradient(source_lz, source_t)
    t_obs = float(source_t[-1])

    return pd.DataFrame(
        {
            "t": source_t,
            "p": source_p,
            "e": source_e,
            "pdot": source_pdot,
            "edot": source_edot,
            "E": source_energy,
            "Lz": source_lz,
            "Edot": source_edot_energy,
            "Lzdot": source_lzdot,
            "Phi_r": source_phi_r,
            "Phi_phi": source_phi_phi,
            "Omega_r": source_omega_r,
            "Omega_phi": source_omega_phi,
            "M": np.full_like(source_t, M),
            "mu": np.full_like(source_t, mu),
            "p0": np.full_like(source_t, p0),
            "e0": np.full_like(source_t, e0),
            "T_obs": np.full_like(source_t, t_obs),
        },
        columns=OUTPUT_COLUMNS,
    )


def write_output(path: Path, params: dict[str, float], df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("# fsi_mode=full\n")
        fh.write("# regenerated_from=Fast_Self-Forced_Inspirals -f\n")
        for key in (
            "z", "theta_S", "phi_S", "theta_K", "phi_K",
            "Phi_phi0", "Phi_r0", "Phi_theta0",
        ):
            fh.write(f"# {key}={params[key]:.10e}\n")
        fh.write(" ".join(OUTPUT_COLUMNS) + "\n")
        df.to_csv(
            fh,
            sep=" ",
            index=False,
            header=False,
            float_format="%.10e",
            lineterminator="\n",
        )


def process_case(case: int) -> str:
    params = case_parameters(case)
    q = params["mu"] / params["M"]
    nit_path = run_nit(params["p0"], params["e0"], q)
    nit_df = load_nit_table(nit_path)
    df = convert_nit_to_output(nit_df, params, load_correction_interpolators())
    output = OUTPUT_ROOT / f"case{case}" / "准确值.txt"
    write_output(output, params, df)
    return f"case{case}: rows={len(df)} output={output}"


def main() -> None:
    global OUTPUT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--count", type=int, default=CASE_COUNT)
    parser.add_argument(
        "--exclude-cases",
        type=int,
        nargs="*",
        default=[],
        help="从随机抽样母集排除这些 case 编号",
    )
    parser.add_argument(
        "--cases",
        type=int,
        nargs="*",
        default=None,
        help="显式指定 case 编号；指定后不再随机抽样",
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.count < 1:
        raise SystemExit("--count must be positive")
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")

    if args.cases is not None and args.cases:
        cases = sorted(set(args.cases))
        if len(cases) != args.count or any(case < 1 or case > 100 for case in cases):
            raise SystemExit("--cases 必须包含恰好 --count 个 1 到 100 的不同编号")
    else:
        excluded = set(args.exclude_cases)
        population = [case for case in range(1, 101) if case not in excluded]
        if len(population) < args.count:
            raise SystemExit("排除后可抽样 case 数量不足")
        cases = sorted(random.Random(args.seed).sample(population, args.count))
    OUTPUT_ROOT = args.output_root
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "selected_cases.txt").write_text(
        f"seed={args.seed}\njobs={args.jobs}\ncases="
        + " ".join(map(str, cases)) + "\n",
        encoding="utf-8",
    )
    print(f"seed={args.seed} jobs={args.jobs} cases={' '.join(map(str, cases))}", flush=True)

    failures = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(process_case, case): case for case in cases}
        for future in as_completed(futures):
            case = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                failures.append((case, repr(exc)))
                print(f"case{case}: FAILED {exc}", flush=True)
    if failures:
        raise SystemExit(f"failed cases: {failures}")


if __name__ == "__main__":
    main()
