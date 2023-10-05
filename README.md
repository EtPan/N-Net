# $\hat{N}$-Net
### Introduction

This is the source code for our paper: [Hyperspectral image denoising via spectral noise distribution bootstrap](https://www.sciencedirect.com/science/article/pii/S0031320323003977).

### Usage

#### 1. Requirements

- Python =3.7 
- torch =1.9.0, torchnet, torchvision
- pickle, tqdm, tensorboardX, scikit-image

#### 2. Data Preparation

- download ICVL hyperspectral image database from [here](http://icvl.cs.bgu.ac.il/hyperspectral/) 

  save the data in *.mat format into your folder

- generate data with synthetic noise for training and validation

  ```python
     # change the data folder first
      python  ./data/datacreate.py
  ```


#### 3. Training

```python
   python main.py -a nnet --dataroot (your data root) --phase train
```

#### 4. Testing

- Testing on Synthetic data or Real HSIs with the pre-trained model

  ```python
      python main.py -a nnet --phase test  -r -rp checkpoints/model_best.pth
  ```

### Citation

If you find this work useful, please cite our paper:

```
@article{Pan2023hypersepctral,
        title = {Hyperspectral image denoising via spectral noise distribution bootstrap},
        author = {Erting Pan and Yong Ma and Xiaoguang Mei and Fan Fan and Jiayi Ma},
        journal = {Pattern Recognition},
        volume = {142},
        pages = {109699},
        year = {2023},
        issn = {0031-3203},
        doi = {https://doi.org/10.1016/j.patcog.2023.109699}
        }
```

### Contact 

Feel free to open an issue if you have any question. You could also directly contact us through email at [panerting@whu.edu.cn](mailto:panerting@whu.edu.cn) (Erting Pan)

