"""Command line: ``btcfe <command>``.

btcfe download                    # BTC/USD daily from stooq.pl -> data/btcusd_d.csv
btcfe eval                        # walk-forward comparison of forecasters
btcfe leakage                     # the three mistakes that make forecasts look good
btcfe s2f                         # Stock-to-Flow, fitted to 2019, scored on 2020+
btcfe all                         # everything above except download; charts in docs/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import data

RESULTS = Path("results")
DOCS = Path("docs")


def _save(name: str, obj: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    clean = {k: v for k, v in obj.items() if not k.startswith("_")}
    (RESULTS / f"{name}.json").write_text(json.dumps(clean, indent=1, default=float), "utf-8")


def cmd_download(args) -> None:
    path = data.download(args.data)
    df = data.load(path)
    print(f"{len(df)} days, {df['date'].min().date()} … {df['date'].max().date()} -> {path}")


def cmd_eval(args) -> dict:
    from . import evaluate, report

    df = data.since(data.load(args.data), args.train_from)
    models = args.models.split(",")
    out = evaluate.run(df, models, args.test_from, refit=args.refit)
    t = out["test"]
    print(
        f"\ntrain {out['train']['from']} … {out['train']['to']}  ·  test {t['from']} … {t['to']} "
        f"({t['days']} trading days, refit every {args.refit})"
    )
    print(
        f"\n{'model':10s} {'MAE $':>9s} {'vs naive':>9s} {'95% CI':>15s} {'DM p':>6s} "
        f"{'direction':>9s} {'long/flat':>10s}"
    )
    for m, r in out["results"].items():
        ci = r.get("mae_vs_naive_ci95")
        ci_s = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—"
        dm = f"{r['dm_p']:.2f}" if "dm_p" in r else "—"
        hit = f"{r['direction_hit']:.1%}" if "direction_hit" in r else "—"
        lf = f"{r['long_flat_return_pct']:.0f}%" if "long_flat_return_pct" in r else "—"
        print(
            f"{m:10s} {r['mae_usd']:9.0f} {r.get('mae_vs_naive', 1.0):9.3f} {ci_s:>15s} "
            f"{dm:>6s} {hit:>9s} {lf:>10s}"
        )
    print(
        f"\n'always up' direction hit rate: {out['always_up_hit']:.1%} · buy & hold: "
        f"{out['buy_and_hold_return_pct']:.0f}% · long/flat is before trading costs"
    )
    _save("eval", out)
    DOCS.mkdir(exist_ok=True)
    report.eval_chart(out, DOCS / "eval.png")
    return out


def cmd_leakage(args) -> dict:
    from . import leakage, report

    out = leakage.run(data.since(data.load(args.data), args.train_from))
    s, p, a = out["same_day_features"], out["price_level_r2"], out["one_shot_arima"]
    print("\n1) did BTC close lower than yesterday?")
    print(f"   same-day features + shuffled split: {s['leaky_accuracy']:.1%}")
    print(
        f"   yesterday's data + chronological split: {s['honest_accuracy']:.1%}"
        f"   (majority class {s['majority_baseline']:.1%})"
    )
    print(
        f"\n2) price levels: R² model {p['model_r2']:.4f} vs naive {p['naive_r2']:.4f}; "
        f"MAE ${p['model_mae']:.0f} vs ${p['naive_mae']:.0f}"
    )
    print(
        f"\n3) one-shot ARIMA(5,1,0) over {a['test_days']} days: RMSE ${a['arima_rmse']:.0f} vs "
        f"last price ${a['last_price_rmse']:.0f}; forecast spans "
        f"${a['forecast_range'][0]:.0f}–${a['forecast_range'][1]:.0f}"
    )
    _save("leakage", out)
    DOCS.mkdir(exist_ok=True)
    report.leakage_chart(out, DOCS / "leakage.png")
    return out


def cmd_s2f(args) -> dict:
    from . import report, s2f

    out = s2f.run(data.load(args.data), fit_until=args.fit_until)
    lo, hi = out["out_of_sample_range_actual_over_model"]
    print(
        f"\nlog(price) = {out['intercept']:.2f} + {out['slope']:.2f} · log(S2F), fitted to "
        f"{out['fit_until']}: R² {out['r2_in_sample']:.2f}"
    )
    print(
        f"out of sample ({out['months_out_of_sample']} months): actual / model median "
        f"{out['out_of_sample_median_actual_over_model']:.2f}, range {lo:.2f}–{hi:.2f}"
    )
    print(
        f"last month: model ${out['model_price_last_month']:,.0f} vs actual "
        f"${out['actual_price_last_month']:,.0f}"
    )
    _save("s2f", out)
    DOCS.mkdir(exist_ok=True)
    report.s2f_chart(out, DOCS / "s2f.png")
    return out


def cmd_all(args) -> None:
    cmd_eval(args)
    cmd_leakage(args)
    cmd_s2f(args)


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(
        prog="btcfe", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", default=str(data.DEFAULT_PATH))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download").set_defaults(func=cmd_download)
    for name, func in (
        ("eval", cmd_eval),
        ("leakage", cmd_leakage),
        ("s2f", cmd_s2f),
        ("all", cmd_all),
    ):
        s = sub.add_parser(name)
        s.add_argument("--train-from", default="2014-01-01")
        s.add_argument("--test-from", default="2023-01-01")
        s.add_argument("--refit", type=int, default=21, help="refit every N trading days")
        s.add_argument(
            "--models",
            default="drift,arima,ridge,boosting",
            help="comma-separated; add lstm with pip install -e '.[lstm]'",
        )
        s.add_argument("--fit-until", default="2019-12-31", help="S2F training cut-off")
        s.set_defaults(func=func)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
