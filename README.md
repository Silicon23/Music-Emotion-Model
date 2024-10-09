# Music-Emotion-Model

This is an implementation [BYOL for Audio (BYOL-A)](https://github.com/nttcslab/byol-a) trained from scratch on [PMEmo-2019](https://www.next.zju.edu.cn/archive/pmemo/) for Music Emotion Recognition (MER) tasks.

**Some important config settings**
```python
sample_rate = 16000 # 16k Hz
unit_sec = 0.95 # unit embedding length for byol-a model
max_sequence_length = 30 # embedding sequence length for transformers layers
```
See more detailed settings in [config.yaml](./byol-a/config.yaml)

# References

```BibTeX
@article{niizumi2023byol-a,
    title={{BYOL for Audio}: Exploring Pre-trained General-purpose Audio Representations},
    author={Niizumi, Daisuke and Takeuchi, Daiki and Ohishi, Yasunori and Harada, Noboru and Kashino, Kunio},
    journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
    publisher={Institute of Electrical and Electronics Engineers (IEEE)},
    year={2023},
    volume={31},
    pages={137–151},
    doi={10.1109/TASLP.2022.3221007},
    url={http://dx.doi.org/10.1109/TASLP.2022.3221007},
    ISSN={2329-9304}
}
```

```BibTeX
@inproceedings{Zhang:2018:PDM:3206025.3206037,
    author = {Zhang, Kejun and Zhang, Hui and Li, Simeng and Yang, Changyuan and Sun, Lingyun},
    title = {The PMEmo Dataset for Music Emotion Recognition},
    booktitle = {Proceedings of the 2018 ACM on International Conference on Multimedia Retrieval},
    series = {ICMR '18},
    year = {2018},
    isbn = {978-1-4503-5046-4},
    location = {Yokohama, Japan},
    pages = {135--142},
    numpages = {8},
    url = {http://doi.acm.org/10.1145/3206025.3206037},
    doi = {10.1145/3206025.3206037},
    acmid = {3206037},
    publisher = {ACM},
    address = {New York, NY, USA},
    keywords = {dataset, eda, experiment, music emotion recognition},
} 
```