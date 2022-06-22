# 🐸💬 TTS French Recipes

For running the recipes

1. Download french data

	You need `parallel`, `pigz`, `pxz`, `rsync` and `ffmpeg` to efficiently convert large datasets. You'll thank me later.

	- M-AILABS dataset can be downloaded either manually from [its official website](https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/) or using ```download_mailabs_fr.sh [PATH_TO_MAILABS]```.
	M-AI Labs' sample rate is 16K Hz which too low for most TTS application. You need to upsample the dataset.
	
	_i.e. with a sample rate of 22.05K Hz._
	```bash
	n=$(nproc)
	sample_rate=22050
	mailabs_path=path/to/M_AILABS/fr_FR
	echo "Converting wav files to ${sample_rate} Hz WAV using ${n} processes." && \
	find . -type f -name "*.wav" -print0 | parallel -0 --eta -j $n mv {} {}_ && ffmpeg -loglevel 0 -n -i {}_ -ar ${sample_rate} -ac 1 {} && \rm {}_ && \
	mv -f $mailabs_path "${mailabs_path}_${sample_rate}" && \
	echo "Archiving dataset to save time." && \
	tar cf - "${mailabs_path}_${sample_rate}" | pigz > "${mailabs_path}_${sample_rate}.zip" && \
	echo "Everthing is done." || \
	echo "Conversion stopped without finishing."
	```
	
	- The Multilingual-LibriSpeech (MLS) dataset can be downloaded on [OpenSLR.org](http://www.openslr.org/94/) or using `download_mls_fr.sh [PATH_TO_MLS]`.
	MLS is formatted using the opus format so we need to convert it to wav.
	
	```bash
	n=$(nproc)
	opus_path=path/to/mls_french_opus
	wav_path=path/to/mls_french_wav
	sample_rate=22050
	echo "Converting opus files to ${sample_rate} Hz WAV using ${n} processes." && \
	find . -type f -name "*.opus" -print0 | parallel -0 --eta -j $n ffmpeg -loglevel 0 -n -i {} -ar ${sample_rate} -ac 1 {.}.wav && \
	echo "Preparing files." && \
	rsync -arh --no-compress --info=progress2 $opus_path $wav_path && \
	echo "Tidying things up." && \
	cd ${opus_path} && find . -name "*.wav" -delete && \
	cd ../../${wav_path} && find . -name "*.opus" -delete && \
	echo "Archiving dataset to save time." && cd ../.. && \
	tar cf - mls_french_wav_${sample_rate} | pxz -c -T0 > mls_french_wav_${sample_rate}.tar.xz && \
	echo "Everything is done." || \
	echo "Conversion stopped without finishing."
	```

	MLS doesn't support ponctuation unfortunately. So we'll use a transformer to restore it.

	Install `deepmultilingualpunctuation` and `sentencepiece` using PIP and run the following script in the parent directory of `mls_french_wav_*`.

	```python3
	#!/usr/bin/env python3

	from deepmultilingualpunctuation import PunctuationModel
	from glob import glob

	model = PunctuationModel()

	mls_path = "mls_french_wav_*/" # Change this for your language
	text_file_list = glob(f"{mls_path}/*/transcripts.txt")

	split_char = "	"

	for text_file_path in text_file_list:
		print(f"Processing {text_file_path}")
		punctuated_transcripts = []
		with open(text_file_path, 'r') as text_file:
			text_list = text_file.readlines()
			for text in text_list:
				s_idx, _text = text.split(split_char)
				clean_text = model.restore_punctuation(_text)
				transcript_line = f"{s_idx}{split_char}{clean_text}"
				punctuated_transcripts.append(transcript_line)
		punctuated_text_file_path = text_file_path.replace(".txt", "_punctuated.txt")
		with open(punctuated_text_file_path, 'w') as f:
			for t in punctuated_transcripts:
				f.write("%s\n" % t)
	```

2. Train a TTS model for your French variant.

    To train AlignTTS on French we need to install `espeak` or `espeak-ng`.

	You can choose between the availible flavors;
	```bash
	❯ espeak --voices | grep fr-
	5  fr-be           --/M      French_(Belgium)   roa/fr-BE            (fr 8)
	5  fr-ch           --/M      French_(Switzerland) roa/fr-CH            (fr 8)
	5  fr-fr           --/M      French_(France)    roa/fr               (fr 5)
	```

	To set `phoneme_language` in the configuration of AlignTTS.

	Then start a training session by distributing the work.

    ```bash
    ❯ python -m trainer.distribute --script recipes/wasertech_fr/align_tts/train_aligntts.py --gpus "0,1"
    ```

3. Train a vocoder

	If you want to change the speaker's voice, you need to point HiFiGAN's `data_path` to a directory containing audio of the same speaker.

	Distribute the load using:
    ```bash
    ❯ python -m trainer.distribute --script recipes/wasertech_fr/hifigan/train_hifigan.py --gpus "0,1"
    ```

💡 Note that these runs are just templates to help you start training your first model. They are not optimized for the best
result. Double-check the configurations and feel free to share your experiments to find better parameters together 💪.
