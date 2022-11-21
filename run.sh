#!/usr/bin/env bash
set -eo pipefail

mkdir -p $MNT_DIR

gcsfuse --debug_gcs --debug_fuse $BUCKET $MNT_DIR

exec uvicorn inference_server:api --host 0.0.0.0 --port $PORT &

wait -n