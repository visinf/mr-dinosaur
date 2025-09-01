#!/usr/bin/env bash
#
# Download TRI-PD (PD_simplified_00..09) from Google Drive with OAuth Access Token.
# Usage:
#   # Way 1: Environment variable
#   export ACCESS_TOKEN="ya29...."
#   ./download_tri_pd.sh -o /path/to/save --remove
#
#   # Way 2: Pass as argument (higher priority than environment variable)
#   ./download_tri_pd.sh -t "ya29..." -o /path/to/save --remove

# access token of google drive
# https://developers.google.com/oauthplayground/?code=4/0AcvDMrAuxHAPckZWJQQlkHJjzQQinCjxfeXjvgbkRjPGsOzklwcCU8K2CoDn1z3bBwyYlQ&scope=https://www.googleapis.com/auth/drive.readonly

#  Step 1: Select & authorize APIs
# https://www.googleapis.com/auth/drive.readonly
#  Step 2: Exchange authorization code for tokens
# Get Access token e.g ya29.a0AS3H6Nzrm-zgneJ4I3mEClq3Zm789iDRWUxdgD-rokWbNWleK_srb2mgNbGpS3-u-DHqNBAJDLAg-Zr4yTh1deUN9M-k8o_KX5hA0HBZ1Sl_hm_AS0pie9VxVPbha273vNDSoZowzTNZkX9yMgFQXf5sq5i-DZLJwsYbN_HaaCgYKAQ0SARUSFQHGX2Mil0-4uxDJYMbld332-Jsrew0175

ACCESS_TOKEN_ARG=""
OUTPUT_DIR="."
REMOVE_AFTER_EXTRACT=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t) ACCESS_TOKEN_ARG="$2"; shift 2 ;;
    -o) OUTPUT_DIR="$2"; shift 2 ;;
    --remove) REMOVE_AFTER_EXTRACT=true; shift ;;
    *) echo "Usage: $0 [-t ACCESS_TOKEN] [-o OUTPUT_DIR] [--remove]"; exit 1 ;;
  esac
done

ACCESS_TOKEN="${ACCESS_TOKEN_ARG:-${ACCESS_TOKEN:-}}"

if [[ -z "${ACCESS_TOKEN}" ]]; then
  echo "[ERROR] ACCESS_TOKEN not provided. Please pass it with -t or export ACCESS_TOKEN=..."
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

declare -A FILES=(
  [1XT-q-3RjqWVRLt_TE2mcj1zZ2Xkwx5QJ]="00"
  [1vm7T4GgbpwRUJU014DYX-l1IMows0sB8]="01"
  [17-U4tu-CGxbBHLlxm6BD6ZK10y_PFvxR]="02"
  [1wXYTNbiCsOIUss50tOstK9fpCg8aEe47]="03"
  [1Q_I69yoP5IXaShbdk7QvqgNHJUez89X_]="04"
  [10YH82zVMF_C0ZSPLoMJAV5nPpNdRXAix]="05"
  [1tvBRx2AXmfF7wjP2AAk-DKqEBuItO9-9]="06"
  [15o6IkITwODtqbnAkrP-gZqZ7YAGDWRx5]="07"
  [1fzrAxk0yoejXWp0ZuSWlhc-OxIQcZlmT]="08"
  [16oRxR46f5SgVXzkfD3mp3TTrM6r-LfIi]="09"
)

API="https://www.googleapis.com/drive/v3/files"

download_one() {
  local file_id="$1"
  local suffix="$2"
  local out="${OUTPUT_DIR}/PD_simplified_${suffix}.tar.gz"
  local url="${API}/${file_id}?alt=media"

  echo ""
  echo "[INFO] Downloading: ${out}"
  for attempt in 1 2 3; do
    if curl -L \
      -H "Authorization: Bearer ${ACCESS_TOKEN}" \
      --fail --show-error --progress-bar \
      -C - \
      "${url}" -o "${out}"; then
      echo "[OK] Downloaded: ${out}"
      extract_one "${out}"
      return 0
    fi
    echo "[WARN] Attempt ${attempt} failed, retrying in 3 seconds..."
    sleep 3
  done

  echo "[ERROR] Failed to download: ${out}"
  return 1
}

extract_one() {
  local file_path="$1"
  echo "[INFO] Extracting: ${file_path}"
  tar -xzf "${file_path}" -C "${OUTPUT_DIR}"
  echo "[OK] Extracted: ${file_path}"

  if $REMOVE_AFTER_EXTRACT; then
    rm -f "${file_path}"
    echo "[INFO] Removed archive: ${file_path}"
  fi
}

# Token validation
echo "[INFO] Checking access permission..."
if ! curl -s -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  "${API}?pageSize=1" >/dev/null; then
  echo "[ERROR] Access Token invalid or expired. Please re-acquire a drive.readonly token."
  exit 1
fi

# Download & extract
for id in "${!FILES[@]}"; do
  download_one "${id}" "${FILES[$id]}"
done

echo ""
echo "[DONE] All files processed in: ${OUTPUT_DIR}"