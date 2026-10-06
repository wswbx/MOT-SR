#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from few.trajectory.inspiral import EMRIInspiral
from few.utils.constants import YRSID_SI


ROOT = Path("/public/home/licm")
SOURCE_ROOT = ROOT / "符号回归_100cases"
OUTPUT_ROOT = ROOT / "测试" / "pn5_reproduce"
DT_SEC = 10.0
ENDPOINT_EPS_SEC = 1.0
SOURCE_MODE = "embedded-pn5"


def orbit_energy(p: float, e: float) -> float:
    return math.sqrt(((p - 2.0 - 2.0 * e) * (p - 2.0 + 2.0 * e)) / (p * (p - 3.0 - e * e)))


def orbit_lz(p: float, e: float) -> float:
    return p / math.sqrt(p - 3.0 - e * e)


def central_derivative(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    out = np.empty_like(ys)
    out[0] = (ys[1] - ys[0]) / (xs[1] - xs[0])
    out[-1] = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    if len(xs) > 2:
        out[1:-1] = (ys[2:] - ys[:-2]) / (xs[2:] - xs[:-2])
    return out


def parse_header_value(lines: list[str], key: str, default: float = 0.0) -> float:
    prefix = f"# {key}="
    for line in lines:
        if line.startswith(prefix):
            return float(line.split("=", 1)[1].strip())
    return default


def find_source_file(case_dir: Path) -> Path:
    target = case_dir / "准确值.txt"
    if target.exists():
        return target
    candidates = sorted(case_dir.glob("FastSelf*_full_evolution.txt"))
    if len(candidates) != 1:
        raise RuntimeError(f"{case_dir} 中未找到唯一源文件")
    return candidates[0]


def load_case_source_file(source_file: Path) -> dict:
    header_lines: list[str] = []
    first_row: list[str] | None = None
    with source_file.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                header_lines.append(stripped)
                continue
            if stripped.startswith("t "):
                continue
            first_row = stripped.split()
            break

    if first_row is None or len(first_row) < 18:
        raise RuntimeError(f"{source_file} 缺少首行数据")

    return {
        "M": float(first_row[13]),
        "mu": float(first_row[14]),
        "p0": float(first_row[15]),
        "e0": float(first_row[16]),
        "T_obs_sec": float(first_row[17]),
        "a": 0.0,
        "x0": 1.0,
        "z": parse_header_value(header_lines, "z"),
        "theta_S": parse_header_value(header_lines, "theta_S"),
        "phi_S": parse_header_value(header_lines, "phi_S"),
        "theta_K": parse_header_value(header_lines, "theta_K"),
        "phi_K": parse_header_value(header_lines, "phi_K"),
        "Phi_phi0": parse_header_value(header_lines, "Phi_phi0"),
        "Phi_r0": parse_header_value(header_lines, "Phi_r0"),
        "Phi_theta0": parse_header_value(header_lines, "Phi_theta0"),
    }


def load_case_source(case_dir: Path, source_mode: str) -> dict:
    if source_mode == "embedded-pn5":
        source_file = case_dir / "PN5.txt"
        if not source_file.exists():
            raise RuntimeError(f"{source_file} 不存在，无法读取嵌入元数据")
    elif source_mode == "current-accuracy":
        source_file = find_source_file(case_dir)
    else:
        raise RuntimeError(f"未知 source_mode={source_mode}")
    return load_case_source_file(source_file)


def build_output_arrays(traj: EMRIInspiral, src: dict) -> dict:
    # FEW fixed stepping can leave an exact multiple of dt one sample short
    # because the internal time conversion lands just below tmax.
    integration_T_sec = src["T_obs_sec"] + ENDPOINT_EPS_SEC
    t_years = integration_T_sec / YRSID_SI
    target_steps = int(integration_T_sec // DT_SEC) + 8
    t_arr, p_arr, e_arr, _x_arr, phi_phi_arr, _phi_theta_arr, phi_r_arr = traj(
        src["M"],
        src["mu"],
        src["a"],
        src["p0"],
        src["e0"],
        src["x0"],
        T=t_years,
        dt=DT_SEC,
        err=1e-8,
        DENSE_STEPPING=True,
        buffer_length=target_steps,
        integrate_backwards=False,
        max_step_size=None,
        Phi_phi0=src["Phi_phi0"],
        Phi_theta0=src["Phi_theta0"],
        Phi_r0=src["Phi_r0"],
    )

    t_arr = np.asarray(t_arr, dtype=float)
    p_arr = np.asarray(p_arr, dtype=float)
    e_arr = np.asarray(e_arr, dtype=float)
    phi_phi_arr = np.asarray(phi_phi_arr, dtype=float)
    phi_r_arr = np.asarray(phi_r_arr, dtype=float)
    if len(t_arr) < 2:
        raise RuntimeError("PN5 轨道点不足")

    eorb_arr = np.array([orbit_energy(float(p), float(e)) for p, e in zip(p_arr, e_arr)], dtype=float)
    lz_arr = np.array([orbit_lz(float(p), float(e)) for p, e in zip(p_arr, e_arr)], dtype=float)
    return {
        "t": t_arr,
        "p": p_arr,
        "e": e_arr,
        "pdot": central_derivative(t_arr, p_arr),
        "edot": central_derivative(t_arr, e_arr),
        "E": eorb_arr,
        "Lz": lz_arr,
        "Edot": central_derivative(t_arr, eorb_arr),
        "Lzdot": central_derivative(t_arr, lz_arr),
        "Phi_r": phi_r_arr,
        "Phi_phi": phi_phi_arr,
        "Omega_r": central_derivative(t_arr, phi_r_arr),
        "Omega_phi": central_derivative(t_arr, phi_phi_arr),
    }


def write_pn5_file(case_dir: Path, src: dict, data: dict) -> Path:
    out_path = case_dir / "PN5.txt"
    t_obs_sec = float(data["t"][-1])
    with out_path.open("w") as handle:
        handle.write(f"# z={src['z']:.10e}\n")
        handle.write(f"# theta_S={src['theta_S']:.10e}\n")
        handle.write(f"# phi_S={src['phi_S']:.10e}\n")
        handle.write(f"# theta_K={src['theta_K']:.10e}\n")
        handle.write(f"# phi_K={src['phi_K']:.10e}\n")
        handle.write(f"# Phi_phi0={src['Phi_phi0']:.10e}\n")
        handle.write(f"# Phi_r0={src['Phi_r0']:.10e}\n")
        handle.write(f"# Phi_theta0={src['Phi_theta0']:.10e}\n")
        handle.write("t p e pdot edot E Lz Edot Lzdot Phi_r Phi_phi Omega_r Omega_phi M mu p0 e0 T_obs\n")
        for i in range(len(data["t"])):
            handle.write(
                f"{data['t'][i]:.10e} {data['p'][i]:.10e} {data['e'][i]:.10e} "
                f"{data['pdot'][i]:.10e} {data['edot'][i]:.10e} "
                f"{data['E'][i]:.10e} {data['Lz'][i]:.10e} "
                f"{data['Edot'][i]:.10e} {data['Lzdot'][i]:.10e} "
                f"{data['Phi_r'][i]:.10e} {data['Phi_phi'][i]:.10e} "
                f"{data['Omega_r'][i]:.10e} {data['Omega_phi'][i]:.10e} "
                f"{src['M']:.10e} {src['mu']:.10e} {src['p0']:.10e} "
                f"{src['e0']:.10e} {t_obs_sec:.10e}\n"
            )
    return out_path


def process_case(case_index: int) -> tuple[int, Path, int]:
    src = load_case_source(SOURCE_ROOT / f"case{case_index}", SOURCE_MODE)
    data = build_output_arrays(EMRIInspiral(func="PN5"), src)
    case_dir = OUTPUT_ROOT / f"case{case_index}"
    case_dir.mkdir(parents=True, exist_ok=True)
    out_path = write_pn5_file(case_dir, src, data)
    return case_index, out_path, len(data["t"])


def main() -> None:
    global SOURCE_ROOT, OUTPUT_ROOT, SOURCE_MODE
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--cases", type=int, nargs="+", required=True)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--source-mode",
        choices=("embedded-pn5", "current-accuracy"),
        default="embedded-pn5",
        help="embedded-pn5 只读取现有 PN5 的输入元数据；current-accuracy 读取当前准确值.txt",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")
    SOURCE_ROOT = args.source_root
    OUTPUT_ROOT = args.output_root
    SOURCE_MODE = args.source_mode
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(process_case, case): case for case in args.cases}
        for future in as_completed(futures):
            case = futures[future]
            try:
                _, path, rows = future.result()
                print(f"case{case}: rows={rows} output={path}", flush=True)
            except Exception as exc:
                print(f"case{case}: FAILED {exc}", flush=True)
                raise


if __name__ == "__main__":
    main()
