#!/bin/bash
# take the scripts's parent's directory to prefix all the output paths.
RUN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo $RUN_DIR
MLS_LANG=french

if [ -z "$1" ]; then
    echo "No argument path to mls supplied"
    echo
    echo "Usage:"
    echo
    echo "(tts-venv) bash download_mls_fr.sh path/to/dir/to/download/mls"
    exit 1
fi

if [ -z $(which tts) ]; then
    echo "TTS binary not found in path. Make sure you are inside tts-venv or that TTS is installed on the system for training."
    exit 1
fi

MLS_PATH="${1}"
mkdir -p $MLS_PATH
# download M-AILABS dataset
python -c "from TTS.utils.downloaders import download_mailabs; download_mailabs(path='${MLS_PATH}', language='${MLS_LANG}')"
exit 0
