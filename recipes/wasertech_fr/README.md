# 🐸💬 TTS M-AILABS Fr Recipes

For running the recipes

1. Download the M-AILABS dataset here either manually from [its official website](https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/) or using ```download_mailabs_fr.sh [PATH_TO_MAILABS]```.
	Depending on the model you choose, you might want to upsample the dataset.
	
	```bash
	sample_rate=22050
	for f in ./recipes/wasertech_fr/M-AILABS/fr_FR/**/*.wav; do \
		mv $f ${f}_ && \
		ffmpeg -i ${f}_ -y -ar $sample_rate -ac 1 ${f} &&
		rm ${f}_; \
	done
	```
2. Go to your desired model folder and run the training.

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
