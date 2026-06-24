# BirdCLEF 2026 — Audio Classification Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

End-to-end machine learning pipeline for the Kaggle BirdCLEF 2026 competition:
bird species identification from passive acoustic monitoring (PAM) audio recordings.

**Author**: Nasir Ali — Centre for Development of Advanced Computing (C-DAC), Noida

## Overview

BirdCLEF 2026 challenges participants to classify hundreds of bird species from 5-second
soundscape clips recorded in natural habitats. This pipeline covers:

- Audio preprocessing (mel-spectrogram extraction, noise reduction)
- CNN/Transformer backbone training (EfficientNet-B4 + BiGRU)
- Test-time augmentation and multi-label threshold tuning
- Submission file generation

## Pipeline

```
Raw .ogg audio
  → Resample to 32 kHz
  → Mel-spectrogram (128 mels, hop=512)
  → Normalise + augment (mixup, SpecAugment)
  → EfficientNet-B4 encoder
  → BiGRU temporal pooling
  → Sigmoid classifier (N species)
  → threshold → label set
```

## Requirements

```bash
pip install torch torchaudio timm librosa pandas scikit-learn
```

## Usage

```bash
# Preprocess
python preprocess.py --data_dir /path/to/train_audio --out_dir spectrograms/

# Train
python train.py --config configs/effnet_b4.yaml --epochs 50

# Predict
python predict.py --checkpoint best.ckpt --test_dir test_soundscapes/ --output submission.csv
```

## Results

| Model | CV Score | LB Score |
|-------|----------|----------|
| EfficientNet-B4 + BiGRU | — | — |

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{nasirali_birdclef2026_pipelin,
  author    = {Nasir Ali},
  title     = {BirdCLEF2026 Pipeline},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/nasir26/BirdCLEF2026_Pipeline},
  note      = {Centre for Development of Advanced Computing (C-DAC), Noida, India}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.\
© 2026 Nasir Ali, C-DAC Noida. All rights reserved.
