# Frequently Asked Questions (FAQ)

- **The method generalizability?**

It was validated in two ways. (1) The model was trained on three datasets and tested on LiTS and three other datasets. Our setting is much more comprehensive than the standard AI evaluation protocol ([Bilic et al., MedIA'23](https://www.sciencedirect.com/science/article/pii/S1361841522003085)). (2) The hyper-parameters of the tumor synthesis were determined on the three datasets (i.e., CHAOS, BTCV, Pancreas-CT), and then the synthetic tumors, created by the same set of hyper-parameters, were mixed up with the real tumors in LiTS (no overlapping with the three datasets) for the visual assessment. Therefore, the proposed method should be generalized to other CT scans.

- **Compare tumor segmentation with LiTS leaderboard performance**

In Table 5, the baseline results (5-fold cross-validation on real tumors) were provided by the winner of the MSD challenge ([Tang et al., CVPR'22](https://openaccess.thecvf.com/content/CVPR2022/papers/Tang_Self-Supervised_Pre-Training_of_Swin_Transformers_for_3D_Medical_Image_Analysis_CVPR_2022_paper.pdf)), whose Task03-Liver was LiTS. Our synthetic tumors achieved similar performance to Tang et al. It is unfair to directly compare results between cross-validation and leaderboard because (1) the data used for testing are different and (2) the leaderboard performance is heavily overfitted by cherry-picking the prediction based on the case-by-case DSC scores. 

- **The generalization to different architectures?** 

This has been tested on the classic U-Net architecture. U-Net trained by synthetic tumors achieved a DSC of 57.3%; U-Net trained by real tumors obtained a DSC of 56.4%. The conclusion is consistent with our paper (using Swin UNETR).

- **Clinical knowledge for new tumor analysis tasks**

It is needed in two aspects. First, the tumor develop process. In our work, the growth pattern of hepatocellular carcinoma (HCC) has a predominantly expansive (rather than invasive) growth pattern. HCC generally spreads by infiltrating adjacent liver parenchyma or vessels, reflecting unclear tumor margins or satellite lesions. Rarely, multicentric HCC might be confused with intrahepatic metastases that with similar size to the main HCC. Besides, the presence of capsule appearance is a specific feature of HCC. Second, the imaging characteristics. For instance, "wash-in" and "wash-out" are evaluated on imaging as a comparison with liver parenchyma, displaying as higher or lower density, so that we can generate HCC with the appropriate HU values. 

- **The accuracy of vessel segmentation** 

Frankly, this performance does not matter in this work. AI achieves a DSC of 52.8% and 52.9% with and without vessel segmentation. Vessel segmentation, collision detection, mass effect, and capsule appearance were not used for model training due to their high computational cost (but used for Visual Turing Test).

- **Does the ratio of real/synt data matter?** 

It matters to some extent based on the table below (104 CT scans in total). But our main contribution is that with no real tumor scans annotated, AI can achieve comparable performance to the fully supervised counterpart.

|  real/synt | 1/9 | 3/7 | 5/5 | 7/3 | 9/1 |
|  ----  | ----  | ----  | ----  | ----  | ----  |
DSC (%) | 51.2 | 55.3 | 56.6 | 51.5 | 53.2

- **Generalizability to other organs?**

A tumor synthesis strategy that is universally effective for a variety of organs is certainly an attractive topic and is the Holy Grail of unsupervised tumor segmentation. However, previous syntheses of tumors, reviewed in Related Works, were designed specifically for a single type of abnormality. We are pioneering in our demonstration that purely training on synthetic tumors can achieve performance that is comparable to training on real liver tumors. Adapting our method to other organs requires a deep understanding of the biology and pathology of the specific tumor. We anticipate that the generalizability of our method can be enhanced through the utilization of automated methods such as GANs, Diffusion Models, and NeRF for generating representative imaging characteristics of various types of tumors in multiple organs.

- **Will increased size of real/synt data further improve the performance?** 

At the moment, all publicly accessible CT scans with annotated liver tumors have been used for training our baseline (ranking \#1 in LiTS/MSD). More annotated data or new architectures are needed to overcome the bottleneck of real-tumor training. In contrast, synthetic data allow us to significantly expand the training data without the need for manual annotation efforts. We are currently training models on 2,000 healthy CT scans (which are easier to collect) incorporating synthetic tumors.

- **Why not train AI using every transformation?** 

We did not use all the proposed transformations because it took 5s to generate a synthetic tumor if using everything, and the performance is similar.

- **Why not use offline generator?** 

It is because the synthetic tumors were not diverse enough if pre-generated and saved to the disc. This led to a downgraded performance: DSC = 43.5% (offline) vs. 52.9% (on-the-fly).

- **Why does the HU value follow Gaussian distribution?** 

This was supported by the statistics reported in [Bilic et al., MedIA'23](https://www.sciencedirect.com/science/article/pii/S1361841522003085). The randomly scattered light quanta in space made the HU intensity in any specific point to follow Gaussian distribution ([Alpert et al., TMI'82](https://ieeexplore.ieee.org/abstract/document/4307561)).
