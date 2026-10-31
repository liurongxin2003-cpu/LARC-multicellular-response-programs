import argparse
import csv
import gzip
import sys
from collections import Counter, defaultdict
from pathlib import Path


sys.stdout.reconfigure(encoding="utf-8")


def clean(value: str) -> str:
    return value.strip().strip('"')


def inspect_series_matrix(path: Path):
    fields = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!Sample_"):
                continue
            row = next(csv.reader([line.rstrip("\n")], delimiter="\t"))
            fields[row[0]].append([clean(x) for x in row[1:]])

    sample_count = len(fields.get("!Sample_geo_accession", [[]])[0])
    print(f"sample_count\t{sample_count}")
    for field, rows in fields.items():
        print(f"\nFIELD\t{field}\trows={len(rows)}")
        if field == "!Sample_characteristics_ch1":
            by_key = defaultdict(Counter)
            for row in rows:
                for value in row:
                    if ":" in value:
                        key, val = value.split(":", 1)
                        by_key[key.strip().lower()][val.strip()] += 1
                    else:
                        by_key["<unkeyed>"][value] += 1
            for key, counts in sorted(by_key.items()):
                print(f"KEY\t{key}")
                for value, count in counts.most_common():
                    print(f"VALUE\t{count}\t{value}")
        else:
            values = Counter(v for row in rows for v in row)
            for value, count in values.most_common(20):
                print(f"VALUE\t{count}\t{value}")


def inspect_family_soft(path: Path):
    sample = None
    samples = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("^SAMPLE = "):
                if sample:
                    samples.append(sample)
                sample = defaultdict(list)
                sample["geo_accession"].append(clean(line.split("=", 1)[1]))
            elif sample is not None and line.startswith("!Sample_"):
                key, value = line.split("=", 1)
                sample[key.replace("!Sample_", "").strip()].append(clean(value))
    if sample:
        samples.append(sample)

    print(f"sample_count\t{len(samples)}")
    keys = Counter(k for sample in samples for k in sample)
    for key in sorted(keys):
        print(f"\nFIELD\t{key}\tsamples={keys[key]}")
        if key == "characteristics_ch1":
            by_key = defaultdict(Counter)
            for sample in samples:
                for value in sample.get(key, []):
                    if ":" in value:
                        item_key, val = value.split(":", 1)
                        by_key[item_key.strip().lower()][val.strip()] += 1
                    else:
                        by_key["<unkeyed>"][value] += 1
            for item_key, counts in sorted(by_key.items()):
                print(f"KEY\t{item_key}")
                for value, count in counts.most_common():
                    print(f"VALUE\t{count}\t{value}")
        else:
            values = Counter(v for sample in samples for v in sample.get(key, []))
            for value, count in values.most_common(20):
                print(f"VALUE\t{count}\t{value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["series-matrix", "family-soft"])
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.kind == "series-matrix":
        inspect_series_matrix(args.path)
    else:
        inspect_family_soft(args.path)


if __name__ == "__main__":
    main()
