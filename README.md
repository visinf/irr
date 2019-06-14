# Iterative Residual Refinement <br/> for Joint Optical Flow and Occlusion Estimation

This repository is the PyTorch implementation of the paper:

**Iterative Residual Refinement for Joint Optical Flow and Occlusion Estimation (CVPR 2019)**  
[Junhwa Hur](https://sites.google.com/site/hurjunhwa) and [Stefan Roth](https://www.visinf.tu-darmstadt.de/team_members/sroth/sroth.en.jsp)  
Department of Computer Science, TU Darmstadt  
[[Preprint]](https://arxiv.org/pdf/1904.05290.pdf) &ensp; [[Proceeding]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Hur_Iterative_Residual_Refinement_for_Joint_Optical_Flow_and_Occlusion_Estimation_CVPR_2019_paper.pdf) &ensp; [[Supplemental]](http://openaccess.thecvf.com/content_CVPR_2019/supplemental/Hur_Iterative_Residual_Refinement_CVPR_2019_supplemental.pdf)


Please cite the paper below if you find our paper and source codes are useful.  

    @inproceedings{Hur:2019:IRR,  
      Author = {Junhwa Hur and Stefan Roth},  
      Booktitle = {CVPR},  
      Title = {Iterative Residual Refinement for Joint Optical Flow and Occlusion Estimation},  
      Year = {2019}  
    }

Contact: Junhwa Hur [fname.lname]@visinf.tu-darmstadt.de

## Getting started
This code has been developed under Anaconda(Python 3.6), Pytorch 0.4.1 and CUDA 8.0 on Ubuntu 16.04.

1. Please install the followings:

   - Anaconda (Python 3.6)
   - __PyTorch 0.4.1__ (Linux, Conda, Python 3.6, CUDA 8.0)
   - tqdm (`conda install -c conda-forge tqdm`)

2. Then, install the correlation package:
   ```
   ./install.sh
   ```

3. The datasets used for this projects are followings:
    - FlyingChairsOcc dataset (available upon request)
    - [MPI Sintel Dataset](http://sintel.is.tue.mpg.de/downloads)
    - [KITTI Optical Flow 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) and [KITTI Optical Flow 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow)
    - [FlyingThings3D subset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

  
## Training

The `scripts` folder contains training scripts of experiments demonstrated in the paper.  
To train the model, you can simply run the script file, e.g., `./IRR-PWC_flyingChairsOcc.sh`.  
In script files, please configure your own experiment directory (EXPERIMENTS_HOME) and dataset directory in your local system (e.g., SINTEL_HOME or KITTI_HOME).


## Pretrained Models

The `saved_check_point` contains the pretrained models of *i)* baseline, *ii)* baseline + irr, and *iii)* full models.  
Additional pretrained models in the ablations study (Table 1 in the main paper) and their training scripts are available upon request.

  
## Inference

The scripts for testing the pre-trained models are located in `scripts/validation`.


## Acknowledgement

Portions of the source code (e.g., training pipeline, runtime, argument parser, and logger) are from [Jochen Gast](https://www.visinf.tu-darmstadt.de/team_members/jgast/jgast.en.jsp)

