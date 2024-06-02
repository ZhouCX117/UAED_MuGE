# MuGE
We add code for MuGE.  
MuGE: Multiple Granularity Edge Detection [pdf](https://www3.cs.stonybrook.edu/~hling/publication-selected.htm)  
Caixia Zhou, Yaping Huang, Mengyang Pu, Qingji Guan, Ruoxi Deng and Haibin Ling  
CVPR2024

# UAED
The Treasure Beneath Multiple Annotations: An Uncertainty-aware Edge Detector  
Caixia Zhou, Yaping Huang, Mengyang Pu, Qingji Guan, Li Huang and Haibin Ling  
CVPR 2023

# Preparing Data
The processed dataset is from LPCB, you can download the used matlab code and processed data from the [Baidu disk](https://pan.baidu.com/s/1F2nAYKsmNxTCI6dmAOGQqg), the code is 3tii.
The complete processed BSDS training dataset can be downloaded from the [Google disk](https://drive.google.com/file/d/1iB2aUKTjDK0URbvUXbXBKBYAROftRKwX/view?usp=sharing).
Training data for the Multicue dataset can be downloaded from the [Quark Disk](https://pan.quark.cn/s/d87cad9abe2e).

# Checkpoint 
BSDS with single scale for UAED: [Quark disk](https://pan.quark.cn/s/9e65e82b3d40) or  [Google disk](https://drive.google.com/file/d/1nv2_TZRyiQh5oU9TnGMzu313OrspD2l5/view?usp=sharing)  
VOC pretrain model for UAED: [Quark disk](https://pan.quark.cn/s/7bfb4fd56242) or [Google disk](https://drive.google.com/file/d/1cfmErOAUgbvMH_MMFsxhc7f_qxxoy01x/view?usp=sharing) 
Pretrain granularity network for MuGE:  [Google disk](https://drive.google.com/file/d/1DBLZvPwI-Z6N70pG8y3-TKWmlUdRjulR/view?usp=drive_link) 
BSDS with scale for MuGE: [Google disk](https://drive.google.com/file/d/15NucsEeHAFwo5O2s11pMR1BikoUtiuUX/view?usp=sharing)  

# Results
UAED Results for BSDS under a single-scale setting can be found [here](https://pan.quark.cn/s/840cd0690997).
# Start
UAED:  
```
python train_uaed.py
```
MuGE:
first download the [checkpoint](https://drive.google.com/file/d/15NucsEeHAFwo5O2s11pMR1BikoUtiuUX/view?usp=sharing), then revise line 39 as the checkpoint path.  
```
python train_muge.py
```
# Best ODS and OIS evaluation for MuGE
1. Run python test_muge.py to obtain the results under different granularities. 
2. Test ODS and OIS for each granularity as normal. 
3. Run eval_muge_best/best_ods_ois.py to obtain the ODS and OIS value. 
4. Run eval_muge_best/select_best_ois_png.py to obtain the selected pictures for best OIS. 
5. Select the threshold for best ODS from the best_ods_0.1/nms-eval/eval_bdry_thr.txt, revise line 8 in eval_muge_best/select_best_ods_png.py and run eval_muge_best/select_best_ods_png.py to select the best pictures for best ODS. 
# Acknowledgement & Citation
The dataset is highly based on the LPCB, and the code is highly based on [RCF_Pytorch_Updated](https://github.com/balajiselvaraj1601/RCF_Pytorch_Updated) and [
segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch). Many thanks for their great work.  
Please consider citing this project in your publications if it helps your research.
```
@inproceedings{zhou2023treasure,
  title={The treasure beneath multiple annotations: An uncertainty-aware edge detector},
  author={Zhou, Caixia and Huang, Yaping and Pu, Mengyang and Guan, Qingji and Huang, Li and Ling, Haibin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15507--15517},
  year={2023}
}
```
```
@inproceedings{zhou2024muge,
  title={MuGE: Multiple Granularity Edge Detection},
  author={Zhou, Caixia and Huang, Yaping and Pu, Mengyang and Guan, Qingji and Deng, Ruoxi and Ling, Haibin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```
```
@inproceedings{deng2018learning,
  title={Learning to predict crisp boundaries},
  author={Deng, Ruoxi and Shen, Chunhua and Liu, Shengjun and Wang, Huibing and Liu, Xinru},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={562--578},
  year={2018}
}
```

