import os, sys

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.align_tts_config import AlignTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.datasets.formatters import mailabs as mailabs_formatter
from TTS.tts.models.align_tts import AlignTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

output_path = "/mnt/Données II/Données/TTS/data/" # os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

mailabs_path = os.path.join(output_path, "extracted/M-AILABS/fr_FR_22.05K/")
mls_path = os.path.join(output_path, "extracted/MLS/mls_french_wav_22.05K/",)

if not os.path.exists(mailabs_path):
    print(f"ERROR: M-AILABS not present in: {mailabs_path}")
    sys.exit(1)

# init configs
mailabs_dataset_config = BaseDatasetConfig(name="mailabs", meta_file_train=None, path=mailabs_path, language="fr_FR", meta_file_val=None)

config = AlignTTSConfig(
    batch_size=16,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=100,
    text_cleaner="french_cleaners",
    use_phonemes=True,
    phoneme_language="fr-fr",
    phoneme_cache_path=os.path.join(output_path, "models/AlignTTS/phoneme_cache"),
    print_step=25,
    print_eval=True,
    mixed_precision=False,
    output_path=os.path.join(output_path, "models/AlignTTS/"),
    datasets=[mailabs_dataset_config],
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# If characters are not defined in the config, default characters are passed to the config
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    mailabs_dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
    formatter=mailabs_formatter
)

# init model
model = AlignTTS(config, ap, tokenizer)

# INITIALIZE THE TRAINER
# Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
# distributed training, etc.
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)

# AND... 3,2,1... 🚀
trainer.fit()
