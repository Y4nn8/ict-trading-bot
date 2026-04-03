"""Tests for CSV data adapter."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from src.market_data.csv_adapter import CSVFormat, load_csv


class TestCSVAdapter:
    """Tests for CSV loading."""

    def _write_csv(self, content: str) -> Path:
        with NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(content)
        return Path(f.name)

    def test_load_generic_csv(self) -> None:
        path = self._write_csv(
            "time,open,high,low,close,volume\n"
            "2024-01-15 10:00:00,1.0800,1.0810,1.0795,1.0805,100\n"
            "2024-01-15 10:05:00,1.0805,1.0815,1.0800,1.0812,120\n"
        )
        df = load_csv(path, CSVFormat.GENERIC)
        assert len(df) == 2
        assert df.columns == ["time", "open", "high", "low", "close", "volume"]
        assert df["open"][0] == pytest.approx(1.0800)

    def test_load_fxcm_csv(self) -> None:
        path = self._write_csv(
            "DateTime,BidOpen,BidHigh,BidLow,BidClose,AskOpen,AskHigh,AskLow,AskClose,TickQty\n"
            "2024-01-15 10:00:00,1.0800,1.0810,1.0795,1.0805,1.0802,1.0812,1.0797,1.0807,150\n"
        )
        df = load_csv(path, CSVFormat.FXCM)
        assert len(df) == 1
        assert df["open"][0] == pytest.approx(1.0800)
        assert df["volume"][0] == 150.0

    def test_load_metatrader_csv(self) -> None:
        path = self._write_csv(
            "Date,Time,Open,High,Low,Close,TickVol,Vol,Spread\n"
            "2024-01-15,10:00:00,1.0800,1.0810,1.0795,1.0805,100,0,8\n"
        )
        df = load_csv(path, CSVFormat.METATRADER)
        assert len(df) == 1
        assert df["close"][0] == pytest.approx(1.0805)

    def test_missing_column_raises(self) -> None:
        path = self._write_csv("time,open,high\n2024-01-15,1.08,1.09\n")
        with pytest.raises(ValueError, match="Missing required column"):
            load_csv(path, CSVFormat.GENERIC)

    def test_no_time_column_raises(self) -> None:
        path = self._write_csv("open,high,low,close\n1.08,1.09,1.07,1.08\n")
        with pytest.raises(ValueError, match="No time column"):
            load_csv(path, CSVFormat.GENERIC)

    def test_sorted_by_time(self) -> None:
        path = self._write_csv(
            "time,open,high,low,close,volume\n"
            "2024-01-15 10:10:00,1.081,1.082,1.080,1.081,100\n"
            "2024-01-15 10:00:00,1.080,1.081,1.079,1.080,100\n"
            "2024-01-15 10:05:00,1.080,1.082,1.079,1.081,100\n"
        )
        df = load_csv(path, CSVFormat.GENERIC)
        times = df["time"].to_list()
        assert times == sorted(times)
