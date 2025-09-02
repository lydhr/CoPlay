# CoPlay
Please cite
```
@INPROCEEDINGS{11134003,
  author={Li, Yin and Liu, Bo and Nandakumar, Rajalakshmi},
  booktitle={2025 34th International Conference on Computer Communications and Networks (ICCCN)}, 
  title={CoPlay: Audio-agnostic Cognitive Scaling for Acoustic Sensing}, 
  year={2025},
  volume={},
  number={},
  pages={1-9},
  keywords={Adaptation models;Wireless sensor networks;Accuracy;Frequency modulation;Acoustics;Sensors;Multiple signal classification;Smart devices;Monitoring;Research and development;acoustic sensing;wireless perception;audio generation},
  doi={10.1109/ICCCN65249.2025.11134003}}

```


#### Download datasets
- For Beethoven dataset, run `data/beethoven/download_archive_preprocess.sh`
- For all other datasets from Youtube, such as pop, piano, podcast datasets, please refer to `data/run.sh`.

#### Run training/validation/testing
The entrypoint is `main.py`. To run a samplery ablation study using `main.py` via Hydra:
`python main.py --multirun signal.type=sine,FMCW model.arch=WaveNet,CNN,RNN`



#### Our coding style follows the trainer of HuggingFace/Transformer
File structure:
```
.

├── main.py
├── run.sh
├── trainer
│   ├── loss.py
│   └── trainer.py
└── utils
    ├── dataset_utils.py
    ├── file_utils.py
    ├── signal_utils.py
    └── utils.py
├── datasets
│   ├── __init__.py
│   ├── base_dataset.py
│   ├── beethoven.py
│   └── youtubeMix.py
├── models
│   ├── __init__.py
│   ├── cnn.py
│   ├── rnn.py
│   ├── unet.py
│   └── wavenet.py
├── conf
│   └── config.yaml
├── README.md
├── requirements.txt
├── environment.yml
├── data
│   ├── beethoven
│   │   ├── download_archive_preprocess.sh
│   │   └── itemlist.txt
│   ├── download-from-youtube.sh
│   └── run.sh
├── saved
│   ├── imgs
│   └── models

```