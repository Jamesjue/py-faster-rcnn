#!/usr/bin/env bash
set -e

source /etc/profile.d/openblas.sh && \
    twistd -n web --port 8080 --wsgi detect-web-server.app
