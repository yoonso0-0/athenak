#!/usr/bin/env python3
"""
Run the sphere_mask visual demos (ad-hoc, not pytest regression tests -- see README.md).

    python3 tst/sphere_mask_demo/run_demo.py implosion       [--athena PATH] [--tlim T]
    python3 tst/sphere_mask_demo/run_demo.py wind_accretion  [--athena PATH] [--tlim T]

Each demo is one input file run several ways through command-line overrides; output goes
to tst/sphere_mask_demo/output_<variant>/ (gitignored). Render with render_implosion.py /
render_wind_accretion.py afterwards. (The gravity demo has its own script, gravity_blast.py.)
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

DEMOS = {
    "implosion": {
        "input": "sphere_mask_implosion_demo.athinput",
        "athena": os.path.join("build", "src", "athena"),
        "variants": {
            "baseline": ["sphere_mask/enabled=false"],
            "dirichlet1": [],
            "dirichlet2": ["sphere_mask/vel1=2.0"],
            "reflecting": ["sphere_mask/bc=reflecting"],
            "absorbing": ["sphere_mask/bc=absorbing"],
        },
    },
    "wind_accretion": {
        # blast.cpp is a UserProblem-only pgen: needs a separate build, see README.md
        "input": "sphere_mask_wind_demo.athinput",
        "athena": os.path.join("build_blast", "src", "athena"),
        "variants": {
            "wind": [],
            "accretion": ["sphere_mask/vel_r=-1.0"],
        },
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("demo", choices=sorted(DEMOS))
    parser.add_argument("--athena", default=None, help="path to the athena executable")
    parser.add_argument("--tlim", type=float, default=None, help="override time/tlim")
    args = parser.parse_args()

    demo = DEMOS[args.demo]
    athena = args.athena or os.path.join(REPO_ROOT, demo["athena"])
    if not os.path.isfile(athena):
        sys.exit(f"athena executable not found at {athena} -- build it first (see README.md) "
                 f"or pass --athena")
    input_file = os.path.join(REPO_ROOT, "inputs", "hydro", demo["input"])

    for name, overrides in demo["variants"].items():
        out_dir = os.path.join(HERE, f"output_{name}")
        os.makedirs(out_dir, exist_ok=True)
        cmd = [athena, "-i", input_file, "-d", out_dir, f"job/basename={name}"] + overrides
        if args.tlim is not None:
            cmd.append(f"time/tlim={args.tlim}")
        print("running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=HERE)


if __name__ == "__main__":
    main()
