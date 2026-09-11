"""Unit tests cho generate_smart_sales_dataset.py"""
import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from llm_finetune_gpt_oos.data.generate_smart_sales_dataset import (
    INTENTS,
    PERSONAS,
    CONTEXTS,
    CONSTRAINTS,
    STAGES,
    PRODUCTS,
    SEED_SCENARIOS,
    SYSTEM_SALES_PROMPT,
    build_record,
    generate_dataset,
    _is_validation,
    _pick,
)


class TestPick:
    def test_returns_correct_element(self):
        values = ["a", "b", "c"]
        elem, cursor = _pick(0, values)
        assert elem == "a"
        assert cursor == 0

    def test_wraps_around(self):
        values = ["a", "b", "c"]
        elem, cursor = _pick(5, values)
        assert elem == "c"  # 5 % 3 == 2
        assert cursor == 1  # 5 // 3 == 1

    def test_cursor_advances(self):
        values = ["x", "y"]
        elem, cursor = _pick(7, values)
        assert elem == "y"  # 7 % 2 == 1
        assert cursor == 3  # 7 // 2 == 3


class TestBuildRecord:
    def test_returns_valid_structure(self):
        record = build_record(0, 42)
        assert "id" in record
        assert "messages" in record
        assert len(record["messages"]) == 3
        assert record["messages"][0]["role"] == "system"
        assert record["messages"][1]["role"] == "user"
        assert record["messages"][2]["role"] == "assistant"

    def test_system_prompt_matches(self):
        record = build_record(0, 42)
        assert record["messages"][0]["content"] == SYSTEM_SALES_PROMPT

    def test_different_indices_produce_different_records(self):
        r1 = build_record(0, 42)
        r2 = build_record(1, 42)
        assert r1["id"] != r2["id"]
        assert r1["messages"][1]["content"] != r2["messages"][1]["content"]

    def test_deterministic(self):
        r1 = build_record(100, 42)
        r2 = build_record(100, 42)
        assert r1["id"] == r2["id"]
        assert r1["messages"] == r2["messages"]

    def test_has_required_metadata(self):
        record = build_record(0, 42)
        assert record["synthetic"] is True
        assert record["vertical"] in ("dien_may", "noi_that")
        assert isinstance(record["product_type"], str)
        assert isinstance(record["intent"], str)
        assert isinstance(record["persona"], str)
        assert isinstance(record["context"], str)


class TestIsValidation:
    def test_zero_ratio_always_false(self):
        assert _is_validation("abcdef01", 0.0) is False

    def test_deterministic(self):
        result1 = _is_validation("abcdef01", 0.1)
        result2 = _is_validation("abcdef01", 0.1)
        assert result1 == result2


class TestSeedScenarios:
    def test_all_scenarios_have_required_fields(self):
        for i, s in enumerate(SEED_SCENARIOS):
            assert "vertical" in s, f"Scenario {i} missing vertical"
            assert "category" in s, f"Scenario {i} missing category"
            assert "user_query" in s, f"Scenario {i} missing user_query"
            assert "assistant_response" in s, f"Scenario {i} missing assistant_response"
            assert len(s["assistant_response"]) > 100, f"Scenario {i} response too short"

    def test_minimum_seed_count(self):
        assert len(SEED_SCENARIOS) >= 8, "Need at least 8 gold seed scenarios"


class TestGenerateDataset:
    def test_creates_output_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_dataset(
                output_dir=tmpdir,
                total_combinatorial=100,
                validation_ratio=0.1,
                seed=42,
            )
            train_file = Path(tmpdir) / "smart_sales_train.jsonl"
            val_file = Path(tmpdir) / "smart_sales_val.jsonl"
            assert train_file.exists()
            assert val_file.exists()

    def test_output_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_dataset(output_dir=tmpdir, total_combinatorial=10)
            train_file = Path(tmpdir) / "smart_sales_train.jsonl"
            with open(train_file, encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    assert "messages" in record
                    msgs = record["messages"]
                    assert len(msgs) == 3
                    assert msgs[0]["role"] == "system"
                    assert msgs[1]["role"] == "user"
                    assert msgs[2]["role"] == "assistant"

    def test_train_val_split_ratio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_dataset(output_dir=tmpdir, total_combinatorial=1000, validation_ratio=0.1)
            train_count = sum(1 for _ in open(Path(tmpdir) / "smart_sales_train.jsonl", encoding="utf-8"))
            val_count = sum(1 for _ in open(Path(tmpdir) / "smart_sales_val.jsonl", encoding="utf-8"))
            total = train_count + val_count
            # Seed scenarios always go to train, so val ratio should be roughly 0.1
            assert val_count / total < 0.15, f"Val ratio too high: {val_count/total:.2f}"
            assert val_count > 0, "Validation set is empty"


class TestDataDiversity:
    """Kiểm tra tính đa dạng của dữ liệu — yếu tố quyết định chất lượng SFT."""

    def test_products_cover_both_verticals(self):
        verticals = {p["vertical"] for p in PRODUCTS}
        assert "dien_may" in verticals
        assert "noi_that" in verticals

    def test_minimum_product_count(self):
        assert len(PRODUCTS) >= 15, "Need at least 15 product types"

    def test_minimum_intent_count(self):
        assert len(INTENTS) >= 10, "Need at least 10 intent types"

    def test_all_products_have_benefit(self):
        for p in PRODUCTS:
            assert "benefit" in p, f"Product {p['product']} missing benefit field"
            assert len(p["benefit"]) > 20, f"Product {p['product']} benefit too short"

    def test_combinatorial_space_is_large(self):
        space = len(PRODUCTS) * len(INTENTS) * len(PERSONAS) * len(CONTEXTS) * len(CONSTRAINTS) * len(STAGES)
        assert space > 1_000_000, f"Combinatorial space too small: {space}"
