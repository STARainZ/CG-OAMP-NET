# CG-OAMP-NET

## Introduction
This repository contains an implementation of the conjugate gradient (CG)-based OAMP-NET [1, 2], which utilizes the CG method to avoid the matrix inverse in OAMP iterations and improves the original OAMP through deep learning [3]. 

## Requirements
- Python (>= 3.6)
- Tensorflow (>=2.3.0)
- numpy (>=1.18.5)
- scipy (>=1.4.1)

## Datasets
WINNER II datasets in [Google drive](https://drive.google.com/drive/folders/1EAeylbdQUQWOUVrinGhK24zBc_JeRzoN?usp=sharing)

Massive MIMO 16x16 Over The Air Channel Information in [download url](https://github.com/syuwei110014/Massive-MIMO-16x16-Over-The-Air-Channel-Information)

## Steps to start
### Step 1. Download the source files.

### Step 2. Download the WINNER-II model from the provided URL and put the data in the 'winner_model' folder.

### Step 3. Run main.py for the simulation specified by the system configurations in it.

For more details, please refer to the readme.txt and the source files in this repository.

## Acknowledgement
We would like to thank Prof. Lei Liu from Japan Advanced Institute of Science and Technology for generally sharing the source code of Memory AMP [4] (a baseline algorithm in our comparison [1]) and discussing the details of the algorithm. The National Sun Yat-sen University Antenna Laboratory led by Prof. K. L. Wong is also highly appreciated for designing the antennas at the prototying platform used in our work [1].

## References
[1] X. Zhou, J. Zhang, C.-W. Syu, C.-K. Wen, J. Zhang, and S. Jin, “Model-driven deep learning-based MIMO-OFDM detector: Design, Simulation, and Experimental Results,” to be published.

[2] X. Zhou, J. Zhang, C.-K. Wen, J. Zhang, and S. Jin, “Model-driven deep learning-based signal detector for CP-free MIMO-OFDM systems,” in Proc. IEEE Int. Conf. on Commun. (ICC) Workshop, Jun. 2021, pp. 1–6.

[3] H. He, C. Wen, S. Jin, and G. Y. Li, “Model-driven deep learning for MIMO detection,” IEEE Trans. Signal Process., vol. 68, pp. 1702–1715, Feb. 2020.

[4] L. Liu, S. Huang, and B. M. Kurkoski, “Memory AMP,” Dec. 2020, [Online] Available: https://arxiv.org/abs/2012.10861.

## Contact
If you have any questions or comments about this work, please feel free to contact xy_zhou@seu.edu.cn
