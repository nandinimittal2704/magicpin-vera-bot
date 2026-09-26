"""
generate_submission.py — Generates submission.jsonl from 30 canonical test pairs

Reads:
- dataset/expanded/test_pairs.json (30 canonical pairs T01-T30)
- dataset/expanded/categories/*.json
- dataset/expanded/merchants/*.json
- dataset/expanded/triggers/*.json
- dataset/expanded/customers/*.json

Outputs:
- submission.jsonl (30 lines of JSONL output for the judge)
"""

import json
from pathlib import Path
from composer import compose


def load_dataset_file(filepath: Path) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    base_dir = Path(__file__).parent
    expanded_dir = base_dir / "dataset" / "expanded"

    if not expanded_dir.exists():
        # Generate expanded dataset if not present
        import subprocess
        subprocess.run(["python", str(base_dir / "dataset" / "generate_dataset.py"), "--out", str(expanded_dir)], check=True)

    test_pairs_file = expanded_dir / "test_pairs.json"
    test_pairs_data = load_dataset_file(test_pairs_file)
    pairs = test_pairs_data.get("pairs", [])

    output_lines = []

    print(f"Generating submission.jsonl for {len(pairs)} test pairs...")

    for pair in pairs:
        test_id = pair["test_id"]
        trigger_id = pair["trigger_id"]
        merchant_id = pair["merchant_id"]
        customer_id = pair.get("customer_id")

        # Load contexts
        # Merchant context
        merchant_file = list(expanded_dir.glob(f"merchants/{merchant_id}.json")) or list(expanded_dir.glob(f"merchants/*{merchant_id}*.json"))
        if not merchant_file:
            print(f"Warning: Merchant {merchant_id} not found!")
            continue
        merchant = load_dataset_file(merchant_file[0])

        # Category context
        cat_slug = merchant.get("category_slug") or "dentists"
        cat_file = expanded_dir / "categories" / f"{cat_slug}.json"
        category = load_dataset_file(cat_file) if cat_file.exists() else {}

        # Trigger context
        trg_file = list(expanded_dir.glob(f"triggers/{trigger_id}.json")) or list(expanded_dir.glob(f"triggers/*{trigger_id}*.json"))
        if not trg_file:
            print(f"Warning: Trigger {trigger_id} not found!")
            continue
        trigger = load_dataset_file(trg_file[0])

        # Customer context (optional)
        customer = None
        if customer_id:
            cust_file = list(expanded_dir.glob(f"customers/{customer_id}.json")) or list(expanded_dir.glob(f"customers/*{customer_id}*.json"))
            if cust_file:
                customer = load_dataset_file(cust_file[0])

        # Compose message
        composed = compose(category, merchant, trigger, customer)

        entry = {
            "test_id": test_id,
            "body": composed.get("body", ""),
            "cta": composed.get("cta", "binary"),
            "send_as": composed.get("send_as", "vera"),
            "suppression_key": composed.get("suppression_key", f"suppress:{test_id}"),
            "rationale": composed.get("rationale", "")
        }

        output_lines.append(entry)

    submission_file = base_dir / "submission.jsonl"
    with open(submission_file, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"Successfully generated {len(output_lines)} lines in {submission_file}")


if __name__ == "__main__":
    main()
