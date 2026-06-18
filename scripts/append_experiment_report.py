import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--body-file", required=True)
    args = parser.parse_args()

    report = Path(args.report)
    body_file = Path(args.body_file)
    body = body_file.read_text(encoding="utf-8")
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write(f"## {args.title}\n\n")
        f.write(body.rstrip())
        f.write("\n")


if __name__ == "__main__":
    main()
