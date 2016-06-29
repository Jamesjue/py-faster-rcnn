# /bin/bash 
set -e
nvidia-docker run -p 21080:8080 --name tpod-detect jamesjue/tpod-detect
