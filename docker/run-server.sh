#!/usr/bin/env bash
set -e

source twistd -n web --port 8080 --wsgi detect-web-server.app
