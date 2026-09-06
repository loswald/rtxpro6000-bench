"""Build a self-contained tarball from the pinned suite and adjacent adapter."""
import argparse
import json
from pathlib import Path
import tarfile
from run import HERE, SUITE, digest, verify_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    args = parser.parse_args()
    verified = verify_suite()
    lock = json.loads((HERE / "suite.lock.json").read_text(encoding="utf-8"))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for relative in sorted(lock["files"]):
            archive.add(SUITE / relative, arcname="evalsuite/" + relative)
        for path in sorted(HERE.iterdir()):
            if path.is_file() and path.suffix in (".py", ".json", ".md"):
                archive.add(path, arcname="capability403/" + path.name)
    print(json.dumps({"path": str(output), "sha256": digest(output), "case_count": verified["case_count"]}, indent=2))


if __name__ == "__main__":
    main()
