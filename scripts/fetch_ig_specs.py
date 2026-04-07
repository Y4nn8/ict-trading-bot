"""Fetch real contract specifications from IG Markets API.

Connects to IG (demo or live) and retrieves dealingRules + instrument
details for every configured instrument.  Prints a YAML snippet that
can replace the instruments section in config/default.yml.

Usage:
    uv run python scripts/fetch_ig_specs.py
"""

from __future__ import annotations

import sys

import yaml
from dotenv import load_dotenv

from src.common.config import load_config
from src.market_data.ig_client import IGClient


def main() -> None:
    """Fetch and display instrument specs from IG API."""
    load_dotenv()
    config = load_config()

    if not config.broker.api_key:
        print("ERROR: broker credentials not set in config/default.yml or env vars")
        sys.exit(1)

    client = IGClient(config.broker)
    print("Connecting to IG Markets...")
    client.connect()

    results = []
    for inst in config.instruments:
        print(f"\nFetching specs for {inst.name} ({inst.epic})...")
        try:
            specs = client.fetch_market_details(inst.epic)
            result = {
                "name": inst.name,
                "epic": inst.epic,
                "asset_class": inst.asset_class,
                "leverage": inst.leverage,
                "ig_name": specs.name,
                "ig_type": specs.instrument_type,
                "value_of_one_pip": specs.value_of_one_pip,
                "one_pip_means": specs.one_pip_means,
                "contract_size": specs.contract_size,
                "lot_size": specs.lot_size,
                "min_deal_size": specs.min_deal_size,
                "min_step_distance": specs.min_step_distance,
                "scaling_factor": specs.scaling_factor,
                "margin_factor": specs.margin_factor,
                "margin_factor_unit": specs.margin_factor_unit,
            }
            results.append(result)

            print(f"  value_of_one_pip: {specs.value_of_one_pip}")
            print(f"  one_pip_means:    {specs.one_pip_means}")
            print(f"  contract_size:    {specs.contract_size}")
            print(f"  min_deal_size:    {specs.min_deal_size}")
            print(f"  min_step:         {specs.min_step_distance}")
            print(f"  scaling_factor:   {specs.scaling_factor}")
            print(f"  margin_factor:    {specs.margin_factor} ({specs.margin_factor_unit})")
        except Exception as e:
            print(f"  ERROR: {e}")

    client.disconnect()

    # Print YAML snippet for config
    print("\n" + "=" * 60)
    print("YAML snippet for config/default.yml instruments section:")
    print("=" * 60)

    yml_instruments = []
    for r in results:
        entry: dict[str, object] = {
            "name": r["name"],
            "epic": r["epic"],
            "asset_class": r["asset_class"],
            "leverage": r["leverage"],
            "min_size": r["min_deal_size"],
            "min_stop_distance": r["min_step_distance"],
            "pip_size": r["one_pip_means"],
            "value_per_point": r["value_of_one_pip"] or 1.0,
        }
        yml_instruments.append(entry)

    print(yaml.dump({"instruments": yml_instruments}, default_flow_style=False))


if __name__ == "__main__":
    main()
