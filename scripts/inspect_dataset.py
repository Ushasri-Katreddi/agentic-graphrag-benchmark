import json
from pathlib import Path
from collections import Counter


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

FILES = {
    "corpus": DATA_DIR / "corpus.jsonl",
    "public_questions": DATA_DIR / "eval_public.jsonl",
    "hidden_questions": DATA_DIR / "eval_hidden.jsonl",
}


def inspect_jsonl(path: Path, sample_size: int = 2) -> None:
    print("\n" + "=" * 80)
    print(f"FILE: {path.name}")
    print("=" * 80)

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return

    record_count = 0
    key_counter = Counter()
    samples = []
    text_lengths = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Invalid JSON on line {line_number}: {exc}")
                continue

            record_count += 1

            if isinstance(record, dict):
                key_counter.update(record.keys())

                if len(samples) < sample_size:
                    samples.append(record)

                for key, value in record.items():
                    if isinstance(value, str):
                        text_lengths.append((key, len(value)))

    print(f"\nRecords: {record_count}")

    print("\nFields:")
    for key, count in key_counter.most_common():
        print(f"  - {key}: {count}/{record_count} records")

    if text_lengths:
        print("\nString field lengths:")
        grouped_lengths = {}

        for key, length in text_lengths:
            grouped_lengths.setdefault(key, []).append(length)

        for key, lengths in grouped_lengths.items():
            average = sum(lengths) / len(lengths)
            maximum = max(lengths)
            minimum = min(lengths)

            print(
                f"  - {key}: "
                f"min={minimum}, "
                f"avg={average:.1f}, "
                f"max={maximum}"
            )

    print("\nSample records:")

    for index, sample in enumerate(samples, start=1):
        print(f"\n--- Sample {index} ---")
        print(json.dumps(sample, indent=2, ensure_ascii=False))


def main() -> None:
    print("=" * 80)
    print("AGENTIC GRAPHRAG HACKATHON - DATASET INSPECTOR")
    print("=" * 80)

    print(f"\nDataset directory:")
    print(DATA_DIR)

    for name, path in FILES.items():
        inspect_jsonl(path)


if __name__ == "__main__":
    main()