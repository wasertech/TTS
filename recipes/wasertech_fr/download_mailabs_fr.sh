#!/bin/bash
# take the scripts's parent's directory to prefix all the output paths.
RUN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo $RUN_DIR
M_AILABS_LANG=fr_FR
M_AILABS_LANGUAGE=french

if [ -z "$1" ]; then
    echo "No argument path to m-ailabs supplied"
    echo
    echo "Usage:"
    echo
    echo "(tts-venv) bash download_mailabs_fr.sh path/to/dir/to/download/mailabs"
    exit 1
fi

if [ -z $(which tts) ]; then
    echo "TTS binary not found in path. Make sure you are inside tts-venv or that TTS is installed on the system for training."
    exit 1
fi

M_AILABS_PATH="${1}"
mkdir -p $M_AILABS_PATH
# download M-AILABS dataset
python -c "from TTS.utils.downloaders import download_mailabs; download_mailabs(path='${M_AILABS_PATH}', language='${M_AILABS_LANGUAGE}')"
exit 0
