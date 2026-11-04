#!/usr/bin/env bash
# Fetch the trained oracles into models/ and verify them.
#
#   bash scripts/download_models.sh                 # everything (~425 MB)
#   bash scripts/download_models.sh protein_canya   # one task
#   bash scripts/download_models.sh --from-local /path/to/grace-models-v1.0.0
#
# The weights live on Zenodo because they are too large for git. Everything the
# repository itself needs to run -- task configs, starting sequences, code --
# is already here.
set -euo pipefail

ZENODO_RECORD="${GRACE_ZENODO_RECORD:-TBD}"     # set once the DOI is minted
BUNDLE="grace-models-v1.0.0"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${GRACE_MODELS:-$ROOT/models}"

mkdir -p "$DEST"

if [[ "${1:-}" == "--from-local" ]]; then
    SRC="${2:?--from-local needs a directory}"
    echo "linking models from $SRC"
    for d in "$SRC"/*/; do ln -sfn "$d" "$DEST/$(basename "$d")"; done
    cp "$SRC/MANIFEST.sha256" "$DEST/" 2>/dev/null || true
    echo "done -> $DEST"
    exit 0
fi

if [[ "$ZENODO_RECORD" == "TBD" ]]; then
    cat >&2 <<'MSG'
The Zenodo DOI has not been minted yet.

If you have the model bundle already:
    bash scripts/download_models.sh --from-local /path/to/grace-models-v1.0.0

Otherwise set the record id once it exists:
    GRACE_ZENODO_RECORD=1234567 bash scripts/download_models.sh
MSG
    exit 1
fi

WHAT="${1:-all}"
URL="https://zenodo.org/records/${ZENODO_RECORD}/files"

if [[ "$WHAT" == "all" ]]; then
    echo "downloading $BUNDLE.tar.gz (~425 MB)"
    curl -fL "$URL/$BUNDLE.tar.gz" -o "$DEST/$BUNDLE.tar.gz"
    tar -xzf "$DEST/$BUNDLE.tar.gz" -C "$DEST" --strip-components=1
    rm "$DEST/$BUNDLE.tar.gz"
else
    echo "downloading models for $WHAT"
    mkdir -p "$DEST/$WHAT"
    curl -fL "$URL/${WHAT}.tar.gz" -o "$DEST/${WHAT}.tar.gz"
    tar -xzf "$DEST/${WHAT}.tar.gz" -C "$DEST"
    rm "$DEST/${WHAT}.tar.gz"
fi

if [[ -f "$DEST/MANIFEST.sha256" ]]; then
    echo "verifying checksums"
    (cd "$DEST" && sha256sum -c --quiet MANIFEST.sha256) && echo "all files verified"
fi
echo "done -> $DEST"
python3 "$ROOT/scripts/optimize.py" --list | grep -E "^[a-z_]+ |status" | paste - - | sed 's/\t/  /'
