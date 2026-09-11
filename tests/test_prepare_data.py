import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import yaml

from llm_finetune_gpt_oos.data.prepare_data import extract_archives, prepare_data


class PrepareDataTest(unittest.TestCase):
    def test_rejects_unsafe_archive_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = root / "raw"
            raw.mkdir()
            with zipfile.ZipFile(raw / "unsafe.zip", "w") as archive:
                archive.writestr("../unsafe.csv", "value\n1\n")
            with self.assertRaises(ValueError):
                extract_archives(raw, root / "staging")

    def test_generates_split_and_removes_pii(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = root / "raw"
            raw.mkdir()
            with (raw / "transactions.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Customer_Name", "Email", "Product_Name", "Quantity", "Unit_Price", "Discount_Applied", "Store_Location"])
                writer.writeheader()
                writer.writerows([{"Customer_Name": "Ada Lovelace", "Email": "ada@example.com", "Product_Name": "Laptop", "Quantity": "2", "Unit_Price": "1200", "Discount_Applied": "10", "Store_Location": "Online"}, {"Customer_Name": "Grace Hopper", "Email": "grace@example.com", "Product_Name": "Monitor", "Quantity": "1", "Unit_Price": "300", "Discount_Applied": "0", "Store_Location": "Store"}])
            config = {"input_dir": str(raw), "extraction_dir": str(root / "staging"), "output_dir": str(root / "processed"), "split": {"train_ratio": 0.5, "seed": 42}, "sampling": {"max_rows_per_table": 100, "examples_per_row": 2}, "quality": {"min_assistant_characters": 40, "redact_pii": True}, "source": {"name": "test", "url": "https://example.test", "license": "test"}}
            config_path = root / "dataset.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            manifest = prepare_data(str(config_path))
            self.assertEqual(manifest["examples"], 4)
            content = "\n".join((root / "processed" / name).read_text(encoding="utf-8") for name in ["train.jsonl", "validation.jsonl"])
            self.assertNotIn("Ada Lovelace", content)
            self.assertNotIn("ada@example.com", content)
            self.assertIn("Laptop", content)
            self.assertEqual(len(json.loads((root / "processed" / "manifest.json").read_text(encoding="utf-8"))["lineage"]), 1)


if __name__ == "__main__":
    unittest.main()
