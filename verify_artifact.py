"""Verify the recorded PEA evidence bundle using only the Python standard library."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "evidence-manifest.json"


def digest(path: Path, mode: str) -> str:
    data = path.read_bytes()
    if mode == "text_lf":
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    elif mode != "raw":
        raise ValueError(f"unsupported hash mode: {mode}")
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    hash_mode = manifest.get("hash_mode", "raw")
    failures = []
    for artifact in manifest["artifacts"]:
        path = ROOT / artifact["path"]
        if not path.is_file():
            failures.append(f"missing: {artifact['path']}")
            continue
        observed = digest(path, artifact.get("hash_mode", hash_mode))
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
