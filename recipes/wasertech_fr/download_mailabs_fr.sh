#!/bin/bash
# take the scripts's parent's directory to prefix all the output paths.
RUN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo $RUN_DIR
M_AILABS_LANG=fr_FR
M_AILABS_LANGUAGE=french
M_AILABS_PATH="${RUN_DIR}/M-AILABS"
mkdir -p $M_AILABS_PATH
# download M-AILABS dataset
python -c "from TTS.utils.downloaders import download_mailabs; download_mailabs(path='${M_AILABS_PATH}', language='${M_AILABS_LANGUAGE}')"
