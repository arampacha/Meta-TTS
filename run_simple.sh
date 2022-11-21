#!/usr/bin/env bash
exec uvicorn inference_server:api --host 0.0.0.0 --port $PORT