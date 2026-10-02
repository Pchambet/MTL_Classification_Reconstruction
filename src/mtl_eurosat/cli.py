"""``mtl-eurosat data | run | collect | report``."""

from __future__ import annotations

import argparse

import torch


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mtl-eurosat", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data", help="decode the committed images into data/interim/ and summarise")
    run = sub.add_parser("run", help="train the multi-seed grid (resumable), then collect")
    run.add_argument("--variants", nargs="*", help="subset of variants (default: all)")
    run.add_argument("--device", default="auto", help="auto (mps if available), cpu, cuda, mps")
    run.add_argument("--epochs", type=int, default=None)
    run.add_argument("--threads", type=int, default=3, help="CPU threads for torch")
    sub.add_parser("collect", help="cached runs -> results/*.csv")
    sub.add_parser("report", help="results -> docs/figures/*.png and site/index.html")
    args = parser.parse_args(argv)

    if args.cmd == "data":
        from mtl_eurosat import data

        images = data.build_cache()
        print(f"in-distribution: {images.x_id.shape}, out-of-distribution: {images.x_ood.shape}")
    elif args.cmd == "run":
        from mtl_eurosat import experiment

        torch.set_num_threads(args.threads)
        experiment.run_grid(args.variants, args.device, args.epochs)
        experiment.collect()
    elif args.cmd == "collect":
        from mtl_eurosat import experiment

        experiment.collect()
    elif args.cmd == "report":
        from mtl_eurosat import report

        report.build()


if __name__ == "__main__":
    main()
