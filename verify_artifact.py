"""Verify the recorded PEA evidence bundle using only the Python standard library."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "evidence-manifest.json"


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures = []
    for artifact in manifest["artifacts"]:
        path = ROOT / artifact["path"]
        if not path.is_file():
            failures.append(f"missing: {artifact['path']}")
            continue
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != artifact["sha256"]:
            failures.append(f"hash mismatch: {artifact['path']} (expected {artifact['sha256']}, observed {observed})")
        else:
            print(f"verified  {artifact['path']}  {observed[:12]}…")
    if failures:
        print("\nEvidence verification failed:\n- " + "\n- ".join(failures))
        return 1
    print(f"\n{manifest['audit_id']}: {len(manifest['artifacts'])} artifacts verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
