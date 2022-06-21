import os, sys

from trainer import Trainer, TrainerArgs

from TTS.utils.audio import AudioProcessor
from TTS.vocoder.configs import HifiganConfig
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN

output_path = "/mnt/Données II/Données/TTS/data/" # os.path.dirname(os.path.abspath(__file__))

if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

voice_wav_path = os.path.join(output_path, "extracted/M-AILABS/fr_FR_22.05K/female/ezwa/monsieur_lecoq/wavs/")

if not os.path.exists(voice_wav_path):
    print(f"ERROR: Path to wav for voice not present in: {voice_wav_path}")
    sys.exit(1)

config = HifiganConfig(
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=5,
    epochs=100,
    seq_len=8192,
    pad_short=2000,
    use_noise_augment=True,
    eval_split_size=10,
    print_step=25,
    print_eval=False,
    mixed_precision=False,
    lr_gen=1e-4,
    lr_disc=1e-4,
    data_path=voice_wav_path,
    output_path=os.path.join(output_path, "models/HiFiGAN/"),
)

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# load training samples
eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size)

# init model
model = GAN(config, ap)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()
