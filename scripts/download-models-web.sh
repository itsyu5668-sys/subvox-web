#!/usr/bin/env bash
# Download the WEB-ONLY models for Subvox Web (~1.55 GB).
# Same URLs as subvox-local/scripts/download-models.sh, minus the
# Apple-Silicon MLX builds and the AV / AV-HuBERT backups (not used by
# this server — see app/paths.py CATALOG). Run from repo root, or in the
# Docker build (WORKDIR /app). Resumable: skips files already present.
set -euo pipefail
cd "$(dirname "$0")/.."

get() {  # url dest
  if [ -s "$2" ]; then echo "OK $2 (already present, $(du -h "$2" | cut -f1))"; return; fi
  mkdir -p "$(dirname "$2")"
  echo "DOWNLOAD $2"
  curl -fL --retry 3 --retry-delay 5 -C - -o "$2.part" "$1" && mv "$2.part" "$2"
}

HF=https://huggingface.co

# 1. Auto-AVSR visual-only, LRS3, 19.1% WER (~956 MB)
get $HF/Amanvir/LRS3_V_WER19.1/resolve/main/model.json models/vsr/model.json
get $HF/Amanvir/LRS3_V_WER19.1/resolve/main/model.pth  models/vsr/model.pth
# 2. Subword RNN language model for beam search (~206 MB)
get $HF/Amanvir/lm_en_subword/resolve/main/model.json  models/lm/model.json
get $HF/Amanvir/lm_en_subword/resolve/main/model.pth   models/lm/model.pth
# 3. SentencePiece tokenizer
get https://github.com/mpc001/auto_avsr/raw/main/spm/unigram/unigram5000.model models/lm/unigram5000.model
# 4. MediaPipe face landmarker (~3.6 MB)
get https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
    models/face_landmarker.task
# 5. Qwen3-0.6B cleanup model, GGUF Q4_K_M (~379 MB)
get $HF/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf models/qwen3-0.6b-gguf/Qwen3-0.6B-Q4_K_M.gguf

echo "All web models present:"
du -sh models
