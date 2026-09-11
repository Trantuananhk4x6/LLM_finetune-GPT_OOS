import json
import tempfile
import unittest
from pathlib import Path

import yaml

from llm_finetune_gpt_oos.data.generate_sales_playbook import generate_sales_playbook


class GenerateSalesPlaybookTest(unittest.TestCase):
    def test_generates_complete_safe_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = {"output_dir": str(root), "count": 200, "validation_ratio": 0.2, "seed": 7}
            config_path = root / "synthetic_sales.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            manifest = generate_sales_playbook(str(config_path))
            train_rows = [json.loads(line) for line in (root / "sales_train.jsonl").read_text(encoding="utf-8").splitlines()]
            validation_rows = [json.loads(line) for line in (root / "sales_validation.jsonl").read_text(encoding="utf-8").splitlines()]
            rows = train_rows + validation_rows
            self.assertEqual(manifest["total_examples"], len(rows))
            self.assertEqual(len({row["id"] for row in rows}), len(rows))
            self.assertEqual({row["vertical"] for row in rows}, {"noi_that", "dien_may"})
            self.assertTrue(all(row["synthetic"] for row in rows))
            self.assertTrue(all("khan hiếm giả" not in row["messages"][-1]["content"].lower() for row in rows))


if __name__ == "__main__":
    unittest.main()
