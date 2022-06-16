# 🐸💬 TTS French Recipes

For running the recipes

1. Download french data

	- M-AILABS dataset can be downloaded either manually from [its official website](https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/) or using ```download_mailabs_fr.sh [PATH_TO_MAILABS]```.
	Mailabs' sample rate is 16K Hz which too low for most TTS application. You need to upsample the dataset.
	
	_i.e. with a sample rate of 22.05K Hz._
	```bash
	n=$(nproc)
	sample_rate=22050
	mailabs_path=path/to/M_AILABS/fr_FR
	echo "Converting wav files to ${sample_rate} Hz WAV using ${n} processes." && \
	find . -type f -name "*.wav" -print0 | parallel -0 --eta -j $n mv {} {}_ && ffmpeg -loglevel 0 -n -i {}_ -ar ${sample_rate} -ac 1 {} && rm {}_ && \
	mv -f $mailabs_path "${mailabs_path}_${sample_rate}" && \
	echo "Archiving dataset to save time." && \
	tar cf - "${mailabs_path}_${sample_rate}" | pigz > "${mailabs_path}_${sample_rate}.zip" && \
	echo "Everthing is done." || \
	echo "Conversion stopped without finishing."
	```

	- MLS dataset can be downloaded on [OpenSLR.org](http://www.openslr.org/94/) or using `download_mls_fr.sh [PATH_TO_MLS]`.
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
	tar cf - "${wav_path}" | pigz > mls_french_wav_${sample_rate}.zip && \
	echo "Everything is done." || \
	echo "Conversion stopped without finishing."
	```

	You need `parallel`, `pigz`, `rsync` and `ffmpeg` to efficiently convert large datasets. You'll thank me later.

2. Navigate to your desired model folder and run the training.

    Running Python files. (Choose the desired GPU ID for your run and set ```CUDA_VISIBLE_DEVICES```)
    ```terminal
    CUDA_VISIBLE_DEVICES="0" python train_modelX.py
    ```

    Running bash scripts.
    ```terminal
    bash run.sh
    ```

💡 Note that these runs are just templates to help you start training your first model. They are not optimized for the best
result. Double-check the configurations and feel free to share your experiments to find better parameters together 💪.
