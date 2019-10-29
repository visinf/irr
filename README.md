# Iterative Residual Refinement <br/> for Joint Optical Flow and Occlusion Estimation

<img src=output.gif>

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

Contact: junhwa.hur[at]visinf.tu-darmstadt.de

## Getting started
This code has been developed under Anaconda(Python 3.6), Pytorch 0.4.1 and CUDA 8.0 on Ubuntu 16.04.

1. Please install the followings:

   - Anaconda (Python 3.6)
   - __PyTorch 0.4.1__ (Linux, Conda, Python 3.6, CUDA 8.0)   
     - For __PyTorch 1.1__, Python 3.7, CUDA >= 9.0, the correlation module needs to be installed accordingly: 
        - Please move all files in `models/correlation_package_cu9` to `models/correlation_package` (overwritting)
        - Depending on your system, configure `-gencode`, `-ccbin`, `cuda-path` in `models/correlation_package/setup.py` accordingly
   - tqdm (`conda install -c conda-forge tqdm`)

2. Then, install the correlation package:
   ```
   ./install.sh
   ```

3. The datasets used for this projects are followings:
    - [FlyingChairsOcc dataset](https://github.com/visinf/irr/tree/master/flyingchairsocc)
    - [FlyingThings3D subset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
    - [MPI Sintel Dataset](http://sintel.is.tue.mpg.de/downloads) + [revised occlusion GT](https://download.visinf.tu-darmstadt.de/data/flyingchairs_occ/occlusions_rev.zip)
    - [KITTI Optical Flow 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) and [KITTI Optical Flow 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow)
    

  
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

