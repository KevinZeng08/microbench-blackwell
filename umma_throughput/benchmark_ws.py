#!/usr/bin/env python3
"""
Benchmark tcgen05.mma.ws throughput (1SM, dense).
Sweeps formats, SS/TS, PTX-legal (M, N), and B-collector discard vs reuse.
"""

import subprocess
import csv
import sys
import os
import argparse

# WS-only format IDs (not the AS bench 0-5 map).
MMA_FORMATS = {
    0: {'name': 'BF16', 'k': 16, 'depths': [16, 32, 64, 128, 256]},
    1: {'name': 'TF32', 'k': 8,  'depths': [16, 32, 64, 128, 256]},
    2: {'name': 'E4M3', 'k': 32, 'depths': [32, 64, 128, 256, 512]},
    3: {'name': 'S8',   'k': 32, 'depths': [32, 64, 128, 256, 512]},
    4: {'name': 'F4',   'k': 64, 'depths': [64, 128, 256, 512, 1024]},
}

WS_M = [32, 64, 128]
WS_N = [64, 128, 256]

CSV_FIELDS = [
    'Op', 'Format', 'ABLayout', 'CTAGroup', 'Collector', 'M', 'N', 'K',
    'PipelineDepth', 'Cycles', 'CyclesPerMMA', 'FLOPsPerCycle',
]


def get_mn_configs():
    return [(m, n) for m in WS_M for n in WS_N]


def run_benchmark(m, n, mma_format, depth, ab_layout, collector, verbose=False):
    fmt_info = MMA_FORMATS[mma_format]
    k = fmt_info['k']
    fmt_name = fmt_info['name']
    mode = 'TS' if ab_layout == 1 else 'SS'
    col_name = 'reuse' if collector == 1 else 'discard'
    label = f"WS {fmt_name} {mode} {col_name} 1SM M={m}, N={n}, depth={depth}"

    clean_cmd = ["make", "clean"]
    build_cmd = [
        "make", "umma_ws_tput.out",
        f"MMA_FORMAT={mma_format}",
        f"MMA_M={m}",
        f"MMA_N={n}",
        f"MMA_K={k}",
        f"MMA_DEPTH={depth}",
        "CTA_GROUP=1",
        f"AB_LAYOUT={ab_layout}",
        f"WS_COLLECTOR={collector}",
    ]

    try:
        subprocess.run(clean_cmd, capture_output=True, check=True)

        if verbose:
            print(f"Building {label}...", file=sys.stderr)
        result = subprocess.run(build_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Build failed for {label}:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return None

        if verbose:
            print(f"Running {label}...", file=sys.stderr)
        result = subprocess.run(["./umma_ws_tput.out"], capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            print(f"Run failed for {label}:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return None

        for line in result.stdout.split('\n'):
            if line.startswith('RESULT,'):
                parts = line.split(',')
                if len(parts) >= 8:
                    M, N, K = int(parts[1]), int(parts[2]), int(parts[3])
                    cycles_per_mma = float(parts[7])
                    flops_per_mma = 2 * M * N * K
                    flops_per_cycle = flops_per_mma / cycles_per_mma if cycles_per_mma > 0 else 0
                    return {
                        'Op': 'WS',
                        'Format': fmt_name,
                        'ABLayout': mode,
                        'CTAGroup': 1,
                        'Collector': col_name,
                        'M': M,
                        'N': N,
                        'K': K,
                        'PipelineDepth': int(parts[4]),
                        'Cycles': int(parts[5]),
                        'CyclesPerMMA': cycles_per_mma,
                        'FLOPsPerCycle': flops_per_cycle,
                    }

        print(f"Could not parse output for {label}", file=sys.stderr)
        print(f"stdout: {result.stdout}", file=sys.stderr)
        return None

    except subprocess.TimeoutExpired:
        print(f"Timeout for {label}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error for {label}: {e}", file=sys.stderr)
        return None


def main():
    fmt_help = ', '.join(f"{k}={v['name']}" for k, v in MMA_FORMATS.items())

    parser = argparse.ArgumentParser(description='Benchmark tcgen05.mma.ws throughput')
    parser.add_argument('formats', nargs='+', type=int, metavar='FORMAT',
                        help=f'WS format IDs: {fmt_help}')
    parser.add_argument('-o', '--output', default='ws_tput.csv',
                        help='Output CSV file')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite CSV instead of appending')
    parser.add_argument('--mode', choices=['ss', 'ts', 'all'], default='all',
                        help='AB layout mode: ss, ts, or all.')
    parser.add_argument('--collector', choices=['discard', 'reuse', 'all'], default='discard',
                        help='B collector: discard (default), reuse (b0 fill/use/lastuse), or all.')
    args = parser.parse_args()

    for f in args.formats:
        if f not in MMA_FORMATS:
            print(f"Error: Invalid format {f}. Valid: {list(MMA_FORMATS.keys())}", file=sys.stderr)
            return 1

    selected_fmts = {k: v for k, v in MMA_FORMATS.items() if k in args.formats}
    ab_layouts = {'ss': [0], 'ts': [1], 'all': [0, 1]}[args.mode]
    collectors = {'discard': [0], 'reuse': [1], 'all': [0, 1]}[args.collector]
    mn_configs = get_mn_configs()

    total_runs = 0
    for fmt_info in selected_fmts.values():
        total_runs += (len(ab_layouts) * len(collectors) *
                       len(mn_configs) * len(fmt_info['depths']))

    print(f"Running {total_runs} WS configurations...", file=sys.stderr)

    file_exists = os.path.exists(args.output) and not args.overwrite
    csv_file = open(args.output, 'a' if file_exists else 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if not file_exists:
        writer.writeheader()
        csv_file.flush()

    result_count = 0
    for fmt_id, fmt_info in selected_fmts.items():
        fmt_name = fmt_info['name']
        k = fmt_info['k']

        for ab_layout in ab_layouts:
            mode = 'TS' if ab_layout == 1 else 'SS'
            for collector in collectors:
                col_name = 'reuse' if collector == 1 else 'discard'
                print(f"\n=== WS {fmt_name} {mode} {col_name} 1SM (K={k}) ===", file=sys.stderr)

                for m, n in mn_configs:
                    for depth in fmt_info['depths']:
                        result = run_benchmark(m, n, mma_format=fmt_id, depth=depth,
                                               ab_layout=ab_layout, collector=collector,
                                               verbose=args.verbose)
                        if result:
                            writer.writerow(result)
                            csv_file.flush()
                            result_count += 1
                            print(f"WS {fmt_name} {mode} {col_name} 1SM M={m:3d}, N={n:3d}, "
                                  f"depth={depth:4d}: {result['CyclesPerMMA']:.4f} cyc/MMA, "
                                  f"{result['FLOPsPerCycle']:.0f} FLOPs/cyc", file=sys.stderr)

    csv_file.close()

    if result_count > 0:
        print(f"\nSaved {result_count} results to {args.output}", file=sys.stderr)
    else:
        print("No successful results", file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
