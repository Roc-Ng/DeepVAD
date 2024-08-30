# Deep Learning for Video Anomaly Detection: A Review

This is the official repository for the paper entitled "**Deep Learning for Video Anomaly Detection: A Review**". 

## 📖 Table of contents

- [Existing Reviews](#reviews)
- [Our Taxonomy](#taxonomy)
  - [1. Semi-Supervised Video Anomaly Detection](#1-Semi-Supervised-Video-Anomaly-Detection)
    - [1.1 Model Input](#11-Model-Input)
      - [1.1.1 RGB](#111-RGB)
      - [1.1.2 Optical Flow](#112-Optical-Flow)
      - [1.1.3 Skeleton](#113-Skeleton)
      - [1.1.4 Hybrid](#114-Hybrid)
    - [1.2 Methodology](#12-Methodology)
      - [1.2.1 Self-Supervised Learning](#121-Self-Supervised-Learning)
      - [1.2.2 One-Class Learning](#122-One-Class-Learning)
      - [1.2.3 Interpretable Learning](#123-Interpretable-Learning)
    - [1.3 Network Architecture](#13-Network-Architecture)
      - [1.3.1 Auto-Encoder](#131-Auto-Encoder)
      - [1.3.2 GAN](#132-GAN)
      - [1.3.3 Diffusion](#133-Diffusion)
    - [1.4 Model Refinement](#14-Model-Refinement)
      - [1.4.1 Pseudo Anomalies](#141-Pseudo-Anomalies)
      - [1.4.2 Memory Bank](#142-Memory-Bank)
    - [1.5 Model Output](#15-Model-Output)
      - [1.5.1 Frame Level](#151-Frame-level)
      - [1.5.2 Pixel Level](#152-Pixel-level)
  - [2. Weakly Supervised Video Anomaly Detection](#2-weakly-supervised-video-anomaly-detection)
    - [2.1 Model Input](#21-Model-Input)
      - [2.1.1 RGB](#211-RGB)
      - [2.1.2 Optical Flow](#212-Optical-Flow)
      - [2.1.3 Audio](#213-Audio)
      - [2.1.4 Text](#214-Text)
      - [2.1.5 Hybrid](#215-Hybrid)
    - [2.2 Methodology](#22-methodology)
      - [2.2.1 One-Stage MIL](#221-One-Stage-MIL)
      - [2.2.2 Two-Stage Self-Training](#222-Two-Stage-Self-Training)
    - [2.3 Refinement Strategy](#23-Refinement-Strategy)
      - [2.3.1 Temporal Modeling](#231-Temporal-Modeling)
      - [2.3.2 Spatio-Temporal Modeling](#232-Spatio-Temporal-Modeling)
      - [2.3.3 MIL-Based Refinement](#233-MIL-Based-Refinement)
      - [2.3.4 Feature Metric Learning](#234-Feature-Metric-Learning)
      - [2.3.5 Knowledge Distillation](#235-Knowledge-Distillation)
      - [2.3.6 Leveraging Large Models](#236-Leveraging-Large-Models)
    - [2.4 Model Output](#24-Model-Output)
      - [2.4.1 Frame Level](#241-Frame-Level)
      - [2.4.2 Pixel Level](#242-Pixel-Level)
  - [3. Fully Supervised Video Anomaly Detection](#3-Fully-Supervised-Video-Anomaly-Detection)
    - [3.1 Appearance Input](#31-Appearance-Input)
    - [3.2 Motion Input](#32-Motion-Input)
    - [3.3 Skeleton Input](#33-Skeleton-Input)
    - [3.4 Audio Input](#34-Audio-Input)
    - [3.5 Hybrid Input](#35-Hybrid-Input)
  - [4. Unsupervised Video Anomaly Detection](#4-Unsupervised-Video-Anomaly-Detection)
    - [4.1 Pseudo Label Based Paradigm](#41-Pseudo-Label-Based-Paradigm)
    - [4.2 Change Detection Based Paradigm](#42-Change-Detection-Based-Paradigm)
    - [4.3 Others](#43-Others)
  - [5. Open-Set Supervised Video Anomaly Detection](#5-Open-Set-Supervised-Video-Anomaly-Detection)
    - [5.1 Open-Set VAD](#51-Open-Set-VAD)
    - [5.2 Few-Shot VAD](#52-Few-Shot-VAD)
- [Performance Comparison](#performance-comparison)
- [Citation](#citation)

## Reviews

| Reference                                                                           | Year | Venue               | Main Focus                                 | Main Categorization                                              | UVAD | WVAD | SVAD | FVAD | OVAD | LVAD | IVAD |
|:----------------------------------------------------------------------------------- |:----:|:-------------------:|:------------------------------------------:|:----------------------------------------------------------------:|:----:|:----:|:----:|:----:|:----:|:----:| ---- |
| [Ramachandra et al.](https://ieeexplore.ieee.org/abstract/document/9271895)         | 2020 | IEEE TPAMI          | Semi-supervised single-scene VAD           | Methodology                                                      | ×    | ×    | √    | ×    | ×    | ×    | ×    |
| [Santhosh et al.](https://dl.acm.org/doi/abs/10.1145/3417989)                       | 2020 | ACM CSUR            | VAD applied on road traffic                | Methodology                                                      | √    | ×    | √    | √    | ×    | ×    | ×    |
| [Nayak et al.](https://www.sciencedirect.com/science/article/pii/S0262885620302109) | 2021 | IMAVIS              | Deep learning driven semi-supervised VAD   | Methodology                                                      | ×    | ×    | √    | ×    | ×    | ×    | ×    |
| [Tran et al.](https://dl.acm.org/doi/abs/10.1145/3544014)                           | 2022 | ACM CSUR            | Semi&weakly supervised VAD                 | Architecture                                                     | ×    | ×    | √    | ×    | ×    | ×    | ×    |
| [Chandrakala et al.](https://link.springer.com/article/10.1007/s10462-022-10258-6)  | 2023 | Artif. Intell. Rev. | Deep model-based one&two-class VAD         | Methodology&Architecture                                         | ×    | √    | √    | √    | ×    | ×    | ×    |
| [Liu et al.](https://dl.acm.org/doi/abs/10.1145/3645101)                            | 2023 | ACM CSUR            | Deep models for semi&weakly supervised VAD | Model Input                                                      | √    | √    | √    | √    | ×    | ×    | ×    |
| Our survey                                                                          | 2024 | -                   | Comprehensive VAD taxonomy and deep models | Methodology, Architecture, Refinement, Model Input, Model Output | √    | √    | √    | √    | √    | √    | √    |

*UVAD=Unsupervised VAD, WVAD=Weakly supervised VAD, SVAD=Semi-supervised VAD, FVAD=Fully supervised VAD, OVAD=Open-set supervised VAD, LVAD: Large-model based VAD, IVAD: Interpretable VAD*

## Taxonomy

## 1. Semi-Supervised Video Anomaly Detection

### 1.1 Model Input

#### 1.1.1 RGB

**Frame-Level RGB**

🗓️ **2016**

- 📄 [ConvAE](https://openaccess.thecvf.com/content_cvpr_2016/html/Hasan_Learning_Temporal_Regularity_CVPR_2016_paper.html):Learning temporal regularity in video sequences, 📰 `CVPR` [code](https://github.com/mhasa004/caffe) [homepage](https://mhasa004.github.io/regularity.html)

🗓️ **2017**

- 📄 [ConvLSTM-AE](https://ieeexplore.ieee.org/abstract/document/8019325/):Remembering history with convolutional LSTM for anomaly detection, 📰 `ICCV` [code](https://github.com/zachluo/convlstm_anomaly_detection)

- 📄 [STAE](https://dl.acm.org/doi/abs/10.1145/3123266.3123451): Spatio-temporal autoencoder for video anomaly detection, 📰 `ACM MM`

- 📄 [AnomalyGAN](https://ieeexplore.ieee.org/abstract/document/8296547): Abnormal event detection in videos using generative adversarial nets, 📰 `ICIP`

🗓️ **2019**

- 📄 [AMC](https://openaccess.thecvf.com/content_ICCV_2019/html/Nguyen_Anomaly_Detection_in_Video_Sequence_With_Appearance-Motion_Correspondence_ICCV_2019_paper.html): Anomaly detection in video sequence with appearance-motion correspondence, 📰 `ICCV`  [code](https://github.com/nguyetn89/Anomaly_detection_ICCV2019)

**Patch-Level RGB**

🗓️ **2015**

- 📄 [AMDN](https://arxiv.org/abs/1510.01553):Learning deep representations of appearance and motion for anomalous event detection, 📰 `BMVC`

🗓️ **2017**

- 📄 [AMDN2](https://www.sciencedirect.com/science/article/abs/pii/S1077314216301618):Detecting anomalous events in videos by learning deep representations of appearance and motion, 📰 `CVIU`

- 📄 [Deep-cascade](https://ieeexplore.ieee.org/abstract/document/7858798):Deep-cascade: Cascading 3d deep neural networks for fast anomaly detection and localization in crowded scenes, 📰 `TIP`

🗓️ **2018**

- 📄 [S$^2$-VAE](https://ieeexplore.ieee.org/abstract/document/8513816):Generative neural networks for anomaly detection in crowded scenes, 📰 `TIFS`

🗓️ **2019**

- 📄 [DeepOC](https://ieeexplore.ieee.org/abstract/document/8825555):A deep one-class neural network for anomalous event detection in complex scenes, 📰 `TNNLS`

🗓️ **2020**

- 📄 [GM-VAE](https://www.sciencedirect.com/science/article/abs/pii/S1077314218302674):Video anomaly detection and localization via gaussian mixture fully convolutional variational autoencoder, 📰 `CVIU`

**Object-Level RGB**

🗓️ **2017**

- 📄 [FRCN](https://openaccess.thecvf.com/content_iccv_2017/html/Hinami_Joint_Detection_and_ICCV_2017_paper.html):Joint detection and recounting of abnormal events by learning deep generic knowledge, 📰 `ICCV`

🗓️ **2019**

- 📄 [ObjectAE](https://openaccess.thecvf.com/content_CVPR_2019/html/Ionescu_Object-Centric_Auto-Encoders_and_Dummy_Anomalies_for_Abnormal_Event_Detection_in_CVPR_2019_paper.html):Object-centric auto-encoders and dummy anomalies for abnormal event detection in video, 📰 `CVPR` [code](https://github.com/fjchange/object_centric_VAD)

🗓️ **2021**

- 📄 [HF$^2$-VAD](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_A_Hybrid_Video_Anomaly_Detection_Framework_via_Memory-Augmented_Flow_Reconstruction_ICCV_2021_paper):A hybrid video anomaly detection framework via memory-augmented flow reconstruction and flow-guided frame prediction, 📰 `ICCV` [code](https://github.com/LiUzHiAn/hf2vad)

🗓️ **2022**

- 📄 [HSNBM](https://dl.acm.org/doi/abs/10.1145/3503161.3548199):Hierarchical scene normality-binding modeling for anomaly detection in surveillance videos, 📰 `ACM MM` [code](https://github.com/baoqianyue/HSNBM)

- 📄 [BDPN](https://ojs.aaai.org/index.php/AAAI/article/view/19898):Comprehensive regularization in a bi-directional predictive network for video anomaly detection, 📰 `AAAI`

- 📄 [ER-VAD](https://dl.acm.org/doi/abs/10.1145/3503161.3548091):Evidential reasoning for video anomaly detection, 📰 `ACM MM`

🗓️ **2023**

- 📄 [HSC](https://openaccess.thecvf.com/content/CVPR2023/html/Sun_Hierarchical_Semantic_Contrast_for_Scene-Aware_Video_Anomaly_Detection_CVPR_2023_paper):Hierarchical semantic contrast for scene-aware video anomaly detection, 📰 `CVPR`[code](https://github.com/shengyangsun/HSC_VAD/)

#### 1.1.2 Optical Flow

**Frame Level**

🗓️ **2018**

- 📄 [FuturePred](https://openaccess.thecvf.com/content_cvpr_2018/html/Liu_Future_Frame_Prediction_CVPR_2018_paper.html):Future frame prediction for anomaly detection–a new baseline, 📰 `CVPR`  [code](code：https://github.com/StevenLiuWen/ano_pred_cvpr2018)

🗓️ **2020**

- 📄 [FSCN](https://www.sciencedirect.com/science/article/abs/pii/S0031320320303186):Fast sparse coding networks for anomaly detection in videos, 📰 `PR` [code](https://github.com/Roc-Ng/FSCN_AnomalyDetection)
  
  🗓️ **2021**

- 📄 [F$^2$PN](https://ieeexplore.ieee.org/abstract/document/9622181):Future frame prediction network for video anomaly detection, 📰 `TPAMI` [code](code：https://github.com/StevenLiuWen/ano_pred_cvpr2018)

- 📄 [AMMC-Net](https://ojs.aaai.org/index.php/AAAI/article/view/16177):Appearance-motion memory consistency network for video anomaly detection, 📰 `AAAI` [code](https://github.com/NjuHaoZhang/AMMCNet_AAAI2021)

🗓️ **2022**

- 📄 [STA-Net](https://ieeexplore.ieee.org/abstract/document/9746822):Learning task-specific representation for video anomaly detection with spatialtemporal attention, 📰 `ICASSP`

🗓️ **2023**

- 📄 [AMSRC](https://ieeexplore.ieee.org/abstract/document/10097199):A video anomaly detection framework based on appearance-motion semantics representation consistency, 📰 `ICASSP`

**Patch Level**

🗓️ **2019**

- 📄 [DeepOC](https://ieeexplore.ieee.org/abstract/document/8825555):A deep one-class neural network for anomalous event detection in complex scenes, 📰 `TNNLS`

🗓️ **2020**

- 📄 [ST-CaAE](https://ieeexplore.ieee.org/abstract/document/9055131):Spatial-temporal cascade autoencoder for video anomaly detection in crowded scenes, 📰 `TMM`

- 📄 [Siamese-Net](https://openaccess.thecvf.com/content_WACV_2020/html/Ramachandra_Learning_a_distance_function_with_a_Siamese_network_to_localize_WACV_2020_paper.html):Learning a distance function with a siamese network to localize anomalies in videos, 📰 `WACV`

**Object Level**

🗓️ **2021**

- 📄 [HF$^2$-VAD](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_A_Hybrid_Video_Anomaly_Detection_Framework_via_Memory-Augmented_Flow_Reconstruction_ICCV_2021_paper):A hybrid video anomaly detection framework via memory-augmented flow reconstruction and flow-guided frame prediction, 📰 `ICCV` [code](https://github.com/LiUzHiAn/hf2vad)

🗓️ **2022**

- 📄 [ER-VAD](https://dl.acm.org/doi/abs/10.1145/3503161.3548091):Evidential reasoning for video anomaly detection, 📰 `ACM MM`

- 📄 [Accurate-Interpretable-VAD](https://arxiv.org/abs/2212.00789):Attribute-based representations for accurate and interpretable video anomaly detection, 📰 `Arxiv`  [code](https://github.com/talreiss/Accurate-Interpretable-VAD)

🗓️ **2023**

- 📄 [AMSRC](https://ieeexplore.ieee.org/abstract/document/10097199):A video anomaly detection framework based on appearance-motion semantics representation consistency, 📰 `ICASSP`

#### 1.1.3 Skeleton

🗓️ **2019**

- 📄 [MPED-RNN](https://openaccess.thecvf.com/content_CVPR_2019/html/Morais_Learning_Regularity_in_Skeleton_Trajectories_for_Anomaly_Detection_in_Videos_CVPR_2019_paper.html):Learning regularity in skeleton trajectories for anomaly detection in videos, 📰 `CVPR`  [code](https://github.com/RomeroBarata/skeleton_based_anomaly_detection)

🗓️ **2020**

- 📄 [GEPC](https://openaccess.thecvf.com/content_CVPR_2020/html/Markovitz_Graph_Embedded_Pose_Clustering_for_Anomaly_Detection_CVPR_2020_paper.html):Graph embedded pose clustering for anomaly detection, 📰 `CVPR`  [code](https://github.com/amirmk89/gepc)

- 📄 [MTTP](https://openaccess.thecvf.com/content_WACV_2020/html/Rodrigues_Multi-timescale_Trajectory_Prediction_for_Abnormal_Human_Activity_Detection_WACV_2020_paper.html):Multi-timescale trajectory prediction for abnormal human activity detection, 📰 `WACV`   [homepage](https://github.com/Rodrigues-Royston/Multi-timescale-Trajectory-Prediction-for-Abnormal-Human-Activity-Detection/tree/master)

🗓️ **2021**

- 📄 [NormalGraph](https://www.sciencedirect.com/science/article/abs/pii/S0925231220317720):Normal graph: Spatial temporal graph convolutional networks based prediction network for skeleton based video anomaly detection, 📰 `Neurocomputing`

- 📄 [HSTGCNN](https://ieeexplore.ieee.org/abstract/document/9645572):A hierarchical spatio-temporal graph convolutional neural network for anomaly detection in videos, 📰 `TCSVT`  [code](code：https://github.com/DivineZeng/A-Hierarchical-Spatio-Temporal-Graph-Convolutional-Neural-Network-for-Anomaly-Detection-in-Videos)

🗓️ **2022**

- 📄 [TSIF](https://ieeexplore.ieee.org/abstract/document/9746420):A two-stream information fusion approach to abnormal event detection in video, 📰 `ICASSP`

- 📄 [STGCAE-LSTM](https://www.sciencedirect.com/science/article/abs/pii/S0925231221018373):Human-related anomalous event detection via spatial-temporal graph convolutional autoencoder with embedded long short-term memory network, 📰 `Neurocomputing`

- 📄 [STGformer](https://dl.acm.org/doi/abs/10.1145/3503161.3548369):Hierarchical graph embedded pose regularity learning via spatiotemporal transformer for abnormal behavior detection, 📰 `ACM MM`

🗓️ **2023**

- 📄 [STG-NF](https://openaccess.thecvf.com/content/ICCV2023/html/Hirschorn_Normalizing_Flows_for_Human_Pose_Anomaly_Detection_ICCV_2023_paper.html):Normalizing flows for human pose anomaly detection, 📰 `ICCV` [code](code：https://github.com/orhir/STG-NF)

- 📄 [MoPRL](https://ieeexplore.ieee.org/abstract/document/10185076):Regularity learning via explicit distribution modeling for skeletal video anomaly detection, 📰 `TCSVT`

- 📄 [MoCoDAD](https://openaccess.thecvf.com/content/ICCV2023/html/Flaborea_Multimodal_Motion_Conditioned_Diffusion_Model_for_Skeleton-based_Video_Anomaly_Detection_ICCV_2023_paper.html):Multimodal motion conditioned diffusion model for skeleton-based video anomaly detection, 📰 `ICCV`  [code](https://github.com/aleflabo/MoCoDAD)

🗓️ **2024**

- 📄 [TrajREC](https://openaccess.thecvf.com/content/WACV2024/html/Stergiou_Holistic_Representation_Learning_for_Multitask_Trajectory_Anomaly_Detection_WACV_2024_paper.html):Holistic representation learning for multitask trajectory anomaly detection, 📰 `WACV`

#### 1.1.4 Hybrid

🗓️ **2018**

- 📄 [FuturePred](https://openaccess.thecvf.com/content_cvpr_2018/html/Liu_Future_Frame_Prediction_CVPR_2018_paper.html):Future frame prediction for anomaly detection–a new baseline, 📰 `CVPR`  [code](code：https://github.com/StevenLiuWen/ano_pred_cvpr2018)

🗓️ **2019**

- 📄 [DeepOC](https://ieeexplore.ieee.org/abstract/document/8825555):A deep one-class neural network for anomalous event detection in complex scenes, 📰 `TNNLS`

🗓️ **2021**

- 📄 [HF$^2$-VAD](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_A_Hybrid_Video_Anomaly_Detection_Framework_via_Memory-Augmented_Flow_Reconstruction_ICCV_2021_paper):A hybrid video anomaly detection framework via memory-augmented flow reconstruction and flow-guided frame prediction, 📰 `ICCV` [code](https://github.com/LiUzHiAn/hf2vad)

🗓️ **2024**

- 📄 [EOGT](https://dl.acm.org/doi/abs/10.1145/3662185):Eogt: Video anomaly detection with enhanced object information and global temporal dependency, 📰 `TOMM`

### 1.2 Methodology

#### 1.2.1 Self-Supervised Learning

**Reconstruction**

🗓️ **2016**

- 📄 [ConvAE](https://openaccess.thecvf.com/content_cvpr_2016/html/Hasan_Learning_Temporal_Regularity_CVPR_2016_paper.html):Learning temporal regularity in video sequences, 📰 `CVPR` [code](https://github.com/mhasa004/caffe) [homepage](https://mhasa004.github.io/regularity.html)

🗓️ **2017**

- 📄 [ConvLSTM-AE](https://ieeexplore.ieee.org/abstract/document/8019325/):Remembering history with convolutional LSTM for anomaly detection, 📰 `ICCV` [code](https://github.com/zachluo/convlstm_anomaly_detection)

🗓️ **2018**

- 📄 [FuturePred](https://openaccess.thecvf.com/content_cvpr_2018/html/Liu_Future_Frame_Prediction_CVPR_2018_paper.html):Future frame prediction for anomaly detection–a new baseline, 📰 `CVPR`  [code](https://github.com/StevenLiuWen/ano_pred_cvpr2018)

- 📄 [S$^2$-VAE](https://ieeexplore.ieee.org/abstract/document/8513816):Generative neural networks for anomaly detection in crowded scenes, 📰 `TIFS`

🗓️ **2019**

- 📄 [AMC](https://openaccess.thecvf.com/content_ICCV_2019/html/Nguyen_Anomaly_Detection_in_Video_Sequence_With_Appearance-Motion_Correspondence_ICCV_2019_paper.html): Anomaly detection in video sequence with appearance-motion correspondence, 📰 `ICCV`  [code](https://github.com/nguyetn89/Anomaly_detection_ICCV2019)

🗓️ **2020**

- 📄 [ClusterAE](https://link.springer.com/chapter/10.1007/978-3-030-58555-6_20):Clustering driven deep autoencoder for video anomaly detection, 📰 `ECCV`

- 📄 [SIGnet](https://ieeexplore.ieee.org/abstract/document/9288937):Anomaly detection with bidirectional consistency in videos, 📰 `TNNLS`

🗓️ **2021**

- 📄 [SSR-AE](https://ieeexplore.ieee.org/abstract/document/9632460):Self-supervision-augmented deep autoencoder for unsupervised visual anomaly detection, 📰 `TCYB`

🗓️ **2023**

- 📄 [MoPRL](https://ieeexplore.ieee.org/abstract/document/10185076):Regularity learning via explicit distribution modeling for skeletal video anomaly detection, 📰 `TCSVT`

**Prediction**

🗓️ **2019**

- 📄 [Attention-driven-loss](https://ieeexplore.ieee.org/abstract/document/8943099):Attention-driven loss for anomaly detection in video surveillance, 📰 `TCSVT`  [code](code：https://github.com/joeyzhouty/Attention-driven-loss)

🗓️ **2020**

- 📄 [Multispace](https://ieeexplore.ieee.org/abstract/document/9266126):Normality learning in multispace for video anomaly detection, 📰 `TCSVT`

🗓️ **2021**

- 📄 [HF$^2$-VAD](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_A_Hybrid_Video_Anomaly_Detection_Framework_via_Memory-Augmented_Flow_Reconstruction_ICCV_2021_paper):A hybrid video anomaly detection framework via memory-augmented flow reconstruction and flow-guided frame prediction, 📰 `ICCV` [code](https://github.com/LiUzHiAn/hf2vad)

- 📄 [AMMC-Net](https://ojs.aaai.org/index.php/AAAI/article/view/16177):Appearance-motion memory consistency network for video anomaly detection, 📰 `AAAI`  [code](https://github.com/NjuHaoZhang/AMMCNet_AAAI2021)

- 📄 [ROADMAP](https://ieeexplore.ieee.org/abstract/document/9446996):Robust unsupervised video anomaly detection by multipath frame prediction, 📰 `TNNLS`

- 📄 [AEP](https://ieeexplore.ieee.org/abstract/document/9346050):Abnormal event detection and localization via adversarial event prediction, 📰 `TNNLS`

🗓️ **2022**

- 📄 [STGformer](https://dl.acm.org/doi/abs/10.1145/3503161.3548369):Hierarchical graph embedded pose regularity learning via spatiotemporal transformer for abnormal behavior detection, 📰 `ACM MM`

- 📄 [OGMRA](https://ieeexplore.ieee.org/abstract/document/9859927):Object-guided and motion-refined attention network for video anomaly detection, 📰 `ICME`

🗓️ **2023**

- 📄 [STGCN](https://ieeexplore.ieee.org/abstract/document/10095170):Spatial-temporal graph convolutional network boosted flow-frame prediction for video anomaly detection, 📰 `ICASSP`

- 📄 [AMP-NET](https://ieeexplore.ieee.org/abstract/document/10203018):Amp-net: Appearance-motion prototype network assisted automatic video anomaly detection system, 📰 `TII`

**Visual Cloze Test**

🗓️ **2020**

- 📄 [VEC](https://dl.acm.org/doi/abs/10.1145/3394171.3413973):Cloze test helps: Effective video anomaly detection via learning to complete video events, 📰 `ACM MM`  [code](https://github.com/yuguangnudt/VEC_VAD)

🗓️ **2023**

- 📄 [USTN-DSC](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Video_Event_Restoration_Based_on_Keyframes_for_Video_Anomaly_Detection_CVPR_2023_paper.html):Video event restoration based on keyframes for video anomaly detection, 📰 `CVPR`

- 📄 [VCC](https://ieeexplore.ieee.org/abstract/document/10197574):Video anomaly detection via visual cloze tests, 📰 `TIFS`

**Jigsaw Puzzles**

🗓️ **2022**

- 📄 [STJP](https://link.springer.com/chapter/10.1007/978-3-031-20080-9_29):Video anomaly detection by solving decoupled spatio-temporal jigsaw puzzles, 📰 `ECCV`  [code](https://github.com/gdwang08/Jigsaw-VAD)

🗓️ **2023**

- 📄 [MPT](https://openaccess.thecvf.com/content/ICCV2023/html/Shi_Video_Anomaly_Detection_via_Sequentially_Learning_Multiple_Pretext_Tasks_ICCV_2023_paper.html):Video anomaly detection via sequentially learning multiple pretext tasks, 📰 `ICCV`

- 📄 [SSMTL++](https://www.sciencedirect.com/science/article/abs/pii/S107731422300036X):Ssmtl++: Revisiting self-supervised multi-task learning for video anomaly detection, 📰 `CVIU`

**Contrastive Learning**

🗓️ **2020**

- 📄 [CAC](https://dl.acm.org/doi/abs/10.1145/3394171.3413529):Cluster attention contrast for video anomaly detection, 📰 `ACM MM`

🗓️ **2021**

- 📄 [TAC-Net](https://ieeexplore.ieee.org/abstract/document/9591368):Abnormal event detection using deep contrastive learning for intelligent video surveillance system, 📰 `TII`

🗓️ **2022**

- 📄 [LSH](https://ieeexplore.ieee.org/abstract/document/9882128):Learnable locality-sensitive hashing for video anomaly detection, 📰 `TCSVT`

**Denoising**

🗓️ **2020**

- 📄 [Adv-AE](https://ieeexplore.ieee.org/abstract/document/9194323):Adversarial 3d convolutional autoencoder for abnormal event detection in videos, 📰 `TMM`

🗓️ **2021**

- 📄 [NM-GAN](https://www.sciencedirect.com/science/article/abs/pii/S0031320321001564):Nm-gan: Noise-modulated generative adversarial network for video anomaly detection, 📰 `PR`

**Deep Sparse Coding**

🗓️ **2017**

- 📄 [Stacked-RNN](https://openaccess.thecvf.com/content_iccv_2017/html/Luo_A_Revisit_of_ICCV_2017_paper.html), A revisit of sparse coding based anomaly detection in stacked RNN framework📰 `ICCV` [code](https://github.com/StevenLiuWen/sRNN_TSC_Anomaly_Detection)

🗓️ **2019**

- 📄 [Anomalynet](https://ieeexplore.ieee.org/abstract/document/8649753):Anomalynet: An anomaly detection network for video surveillance, 📰 `TIFS` [code](https://github.com/joeyzhouty/AnomalyNet)

- 📄 [sRNN-AE](https://ieeexplore.ieee.org/abstract/document/8851288):Video anomaly detection with sparse coding inspired deep neural networks, 📰 `TPAMI` [code](https://github.com/StevenLiuWen/sRNN_TSC_Anomaly_Detection)

🗓️ **2020**

- 📄 [FSCN](https://www.sciencedirect.com/science/article/abs/pii/S0031320320303186):Fast sparse coding networks for anomaly detection in videos, 📰 `PR` [code](https://github.com/Roc-Ng/FSCN_AnomalyDetection)

**Patch Inpainting**

🗓️ **2021**

- 📄 [RIAD](https://www.sciencedirect.com/science/article/abs/pii/S0031320320305094):Reconstruction by inpainting for visual anomaly detection, 📰 `PR`  [code](https://github.com/plutoyuxie/Reconstruction-by-inpainting-for-visual-anomaly-detection)

🗓️ **2022**

- 📄 [SSPCAB](https://openaccess.thecvf.com/content/CVPR2022/html/Ristea_Self-Supervised_Predictive_Convolutional_Attentive_Block_for_Anomaly_Detection_CVPR_2022_paper.html):Self-supervised predictive convolutional attentive block for anomaly detection, 📰 `CVPR` [code](https://github.com/ristea/sspcab)

🗓️ **2023**

- 📄 [SSMCTB](https://ieeexplore.ieee.org/abstract/document/10273635):Self-supervised masked convolutional transformer block for anomaly detection, 📰 `TPAMI`  [code](https://github.com/ristea/ssmctb)

🗓️ **2024**

- 📄 [AED-MAE](https://openaccess.thecvf.com/content/CVPR2024/html/Ristea_Self-Distilled_Masked_Auto-Encoders_are_Efficient_Video_Anomaly_Detectors_CVPR_2024_paper.html):Self-distilled masked auto-encoders are efficient video anomaly detectors, 📰 `CVPR` [code](https://github.com/ristea/aed-mae)

**Multiple Task**

🗓️ **2017**

- 📄 [STAE](https://dl.acm.org/doi/abs/10.1145/3123266.3123451): Spatio-temporal autoencoder for video anomaly detection, 📰 `ACM MM`

🗓️ **2019**

- 📄 [MPED-RNN](https://openaccess.thecvf.com/content_CVPR_2019/html/Morais_Learning_Regularity_in_Skeleton_Trajectories_for_Anomaly_Detection_in_Videos_CVPR_2019_paper.html):Learning regularity in skeleton trajectories for anomaly detection in videos, 📰 `CVPR`

- 📄 [AnoPCN](https://dl.acm.org/doi/abs/10.1145/3343031.3350899):Anopcn: Video anomaly detection via deep predictive coding network, 📰 `ACM MM`

🗓️ **2021**

- 📄 [Multitask](https://openaccess.thecvf.com/content/CVPR2021/html/Georgescu_Anomaly_Detection_in_Video_via_Self-Supervised_and_Multi-Task_Learning_CVPR_2021_paper.html):Anomaly detection in video via self-supervised and multi-task learning, 📰 `CVPR`   [homepage](https://github.com/lilygeorgescu/AED-SSMTL)

🗓️ **2022**

- 📄 [HSNBM](https://dl.acm.org/doi/abs/10.1145/3503161.3548199):Hierarchical scene normality-binding modeling for anomaly detection in surveillance videos, 📰 `ACM MM` [code](https://github.com/baoqianyue/HSNBM)

- 📄 [LSH](https://ieeexplore.ieee.org/abstract/document/9882128):Learnable locality-sensitive hashing for video anomaly detection, 📰 `TCSVT`

- 📄 [AMAE](https://ieeexplore.ieee.org/abstract/document/9739751):Appearance-motion united auto-encoder framework for video anomaly detection, 📰 `TCAS-II`

- 📄 [STM-AE](https://ieeexplore.ieee.org/abstract/document/9859727):Learning appearance-motion normality for video anomaly detection, 📰 `ICME`

- 📄 [SSAGAN](https://ieeexplore.ieee.org/abstract/document/9749781):Self-supervised attentive generative adversarial networks for video anomaly detection, 📰 `TNNLS`

🗓️ **2023**

- 📄 [MPT](https://openaccess.thecvf.com/content/ICCV2023/html/Shi_Video_Anomaly_Detection_via_Sequentially_Learning_Multiple_Pretext_Tasks_ICCV_2023_paper.html):Video anomaly detection via sequentially learning multiple pretext tasks, 📰 `ICCV`

- 📄 [SSMTL++](https://www.sciencedirect.com/science/article/abs/pii/S107731422300036X):Ssmtl++: Revisiting self-supervised multi-task learning for video anomaly detection, 📰 `CVIU`

🗓️ **2024**

- 📄 [MGSTRL](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Multi-Scale_Video_Anomaly_Detection_by_Multi-Grained_Spatio-Temporal_Representation_Learning_CVPR_2024_paper.html):Multi-scale video anomaly detection by multi-grained spatiotemporal representation learning, 📰 `CVPR`

#### 1.2.2 One-Class Learning

**One-Class Classifier**

🗓️ **2015**

- 📄 [AMDN](https://arxiv.org/abs/1510.01553):Learning deep representations of appearance and motion for anomalous event detection, 📰 `BMVC`

🗓️ **2018**

- 📄 [Deep SVDD](https://proceedings.mlr.press/v80/ruff18a.html):Deep one-class classification, 📰 `PMLR`  [code](https://github.com/lukasruff/Deep-SVDD-PyTorch)

🗓️ **2019**

- 📄 [DeepOC](https://ieeexplore.ieee.org/abstract/document/8825555):A deep one-class neural network for anomalous event detection in complex scenes, 📰 `TNNLS`
- 📄 [GODS](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9008531):Gods: Generalized one-class discriminative subspaces for anomaly detection,  📰 `ICCV`

🗓️ **2021**

- 📄 [FCDD](https://openreview.net/forum?id=A5VV3UyIQz):Explainable deep one-class classification, 📰 `ICLR`  [code](https://github.com/liznerski/fcdd)

**Gaussian Classifier**

🗓️ **2018**

- 📄 [Deep-anomaly](https://www.sciencedirect.com/science/article/abs/pii/S1077314218300249):Deep-anomaly: Fully convolutional neural network for fast anomaly detection in crowded scenes, 📰 `CVIU`

🗓️ **2020**

- 📄 [GM-VAE](https://www.sciencedirect.com/science/article/abs/pii/S1077314218302674):Video anomaly detection and localization via gaussian mixture fully convolutional variational autoencoder, 📰 `CVIU`

🗓️ **2021**

- 📄 [Deep-cascade](https://ieeexplore.ieee.org/abstract/document/7858798):Deep-cascade: Cascading 3d deep neural networks for fast anomaly detection and localization in crowded scenes, 📰 `TIP`

**Adversarial Classifier**

🗓️ **2018**

- 📄 [ALOCC](https://openaccess.thecvf.com/content_cvpr_2018/html/Sabokrou_Adversarially_Learned_One-Class_CVPR_2018_paper.html):Adversarially learned one-class classifier for novelty detection, 📰 `CVPR`  [code](https://github.com/khalooei/ALOCC-CVPR2018)

- 📄 [AVID](https://link.springer.com/chapter/10.1007/978-3-030-20876-9_31):Avid: Adversarial visual irregularity detection, 📰 `ACCV` [code](https://github.com/cross32768/AVID-Adversarial-Visual-Irregularity-Detection/tree/master)

🗓️ **2020**

- 📄 [ALOCC2](https://ieeexplore.ieee.org/abstract/document/9059022):Deep end-to-end one-class classifier, 📰 `TNNLS`

- 📄 [OGNet](https://openaccess.thecvf.com/content_CVPR_2020/html/Zaheer_Old_Is_Gold_Redefining_the_Adversarially_Learned_One-Class_Classifier_Training_CVPR_2020_paper.html):Old is gold: Redefining the adversarially learned one-class classifier training paradigm, 📰 `CVPR`  [code](https://github.com/xaggi/OGNet)

🗓️ **2022**

- 📄 [OGNet+](https://ieeexplore.ieee.org/abstract/document/9887825):Stabilizing adversarially learned one-class novelty detection using pseudo anomalies, 📰 `TIP`

#### 1.2.3 Interpretable Learning

🗓️ **2017**

- 📄 [FRCN](https://openaccess.thecvf.com/content_iccv_2017/html/Hinami_Joint_Detection_and_ICCV_2017_paper.html):Joint detection and recounting of abnormal events by learning deep generic knowledge, 📰 `ICCV`

🗓️ **2022**

- 📄 [Accurate-Interpretable-VAD](https://arxiv.org/abs/2212.00789):Attribute-based representations for accurate and interpretable video anomaly detection, 📰 `Arxiv`  [code](https://github.com/talreiss/Accurate-Interpretable-VAD)

🗓️ **2023**

- 📄 [InterVAD](https://openaccess.thecvf.com/content/WACV2023/html/Doshi_Towards_Interpretable_Video_Anomaly_Detection_WACV_2023_paper.html):Towards interpretable video anomaly detection, 📰 `WACV`

- 📄 [EVAL](https://openaccess.thecvf.com/content/CVPR2023/html/Singh_EVAL_Explainable_Video_Anomaly_Localization_CVPR_2023_paper.html):Eval: Explainable video anomaly localization, 📰 `CVPR`

🗓️ **2024**

- 📄 [AnomalyRuler](https://openaccess.thecvf.com/content/CVPR2023/html/Singh_EVAL_Explainable_Video_Anomaly_Localization_CVPR_2023_paper.html):Follow the rules: Reasoning for video anomaly detection with large language models, 📰 `ECCV` [code](https://github.com/Yuchen413/AnomalyRuler)

### 1.3 Network Architecture

#### 1.3.1 Auto-Encoder

🗓️ **2016**

- 📄 [Conv-LSTM](https://arxiv.org/abs/1612.00390):Anomaly detection in video using predictive convolutional long short-term memory networks, 📰 `Arxiv`

🗓️ **2017**

- 📄 [STAE](https://dl.acm.org/doi/abs/10.1145/3123266.3123451): Spatio-temporal autoencoder for video anomaly detection, 📰 `ACM MM`

- 📄 [ConvLSTM-AE](https://ieeexplore.ieee.org/abstract/document/8019325/):Remembering history with convolutional LSTM for anomaly detection, 📰 `ICCV` [code](https://github.com/zachluo/convlstm_anomaly_detection)

🗓️ **2019**

- 📄 [DeepOC](https://ieeexplore.ieee.org/abstract/document/8825555):A deep one-class neural network for anomalous event detection in complex scenes, 📰 `TNNLS`

- 📄 [sRNN-AE](https://ieeexplore.ieee.org/abstract/document/8851288):Video anomaly detection with sparse coding inspired deep neural networks, 📰 `TPAMI`

- 📄 [MPED-RNN](https://openaccess.thecvf.com/content_CVPR_2019/html/Morais_Learning_Regularity_in_Skeleton_Trajectories_for_Anomaly_Detection_in_Videos_CVPR_2019_paper.html):Learning regularity in skeleton trajectories for anomaly detection in videos, 📰 `CVPR`

🗓️ **2021**

- 📄 [NormalGraph](https://www.sciencedirect.com/science/article/abs/pii/S0925231220317720):Normal graph: Spatial temporal graph convolutional networks based prediction network for skeleton based video anomaly detection, 📰 `Neurocomputing`

🗓️ **2022**

- 📄 [STGCAE-LSTM](https://www.sciencedirect.com/science/article/abs/pii/S0925231221018373):Human-related anomalous event detection via spatial-temporal graph convolutional autoencoder with embedded long short-term memory network, 📰 `Neurocomputing`

🗓️ **2023**

- 📄 [USTN-DSC](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Video_Event_Restoration_Based_on_Keyframes_for_Video_Anomaly_Detection_CVPR_2023_paper.html):Video event restoration based on keyframes for video anomaly detection, 📰 `CVPR`

🗓️ **2024**

- 📄 [AED-MAE](https://openaccess.thecvf.com/content/CVPR2024/html/Ristea_Self-Distilled_Masked_Auto-Encoders_are_Efficient_Video_Anomaly_Detectors_CVPR_2024_paper.html):Self-distilled masked auto-encoders are efficient video anomaly detectors, 📰 `CVPR`  [code](https://github.com/ristea/aed-mae)

#### 1.3.2 GAN

🗓️ **2018**

- 📄 [FuturePred](https://openaccess.thecvf.com/content_cvpr_2018/html/Liu_Future_Frame_Prediction_CVPR_2018_paper.html):Future frame prediction for anomaly detection–a new baseline, 📰 `CVPR`  [code](https://github.com/StevenLiuWen/ano_pred_cvpr2018)
- 📄 [ALOCC](https://openaccess.thecvf.com/content_cvpr_2018/html/Sabokrou_Adversarially_Learned_One-Class_CVPR_2018_paper.html):Adversarially learned one-class classifier for novelty detection, 📰 `CVPR`  [code](https://github.com/khalooei/ALOCC-CVPR2018)

🗓️ **2019**

- 📄 [AD-VAD](https://ieeexplore.ieee.org/abstract/document/8658774):Training adversarial discriminators for cross-channel abnormal event detection in crowds, 📰 `WACV`

- 📄 [VAD-GAN](https://ojs.aaai.org/index.php/AAAI/article/view/4456):Robust anomaly detection in videos using multilevel representations, 📰 `AAAI`  [code](https://github.com/SeaOtter/vad_gan)

- 📄 [Ada-Net](https://ieeexplore.ieee.org/abstract/document/8892741):Learning normal patterns via adversarial attention-based autoencoder for abnormal event detection in videos, 📰 `TMM`

- 

🗓️ **2020**

- 📄 [OGNet](https://openaccess.thecvf.com/content_CVPR_2020/html/Zaheer_Old_Is_Gold_Redefining_the_Adversarially_Learned_One-Class_Classifier_Training_CVPR_2020_paper.html):Old is gold: Redefining the adversarially learned one-class classifier training paradigm, 📰 `CVPR`  [code](https://github.com/xaggi/OGNet)

🗓️ **2021**

- 📄 [CT-D2GAN](https://dl.acm.org/doi/abs/10.1145/3474085.3475693):Convolutional transformer based dual discriminator generative adversarial networks for video anomaly detection, 📰 `ACM MM`

#### 1.3.3 Diffusion

🗓️ **2023**

- 📄 [FPDM](https://openaccess.thecvf.com/content/ICCV2023/html/Yan_Feature_Prediction_Diffusion_Model_for_Video_Anomaly_Detection_ICCV_2023_paper.html):Feature prediction diffusion model for video anomaly detection, 📰 `ICCV`

- 📄 [MoCoDAD](https://openaccess.thecvf.com/content/ICCV2023/html/Flaborea_Multimodal_Motion_Conditioned_Diffusion_Model_for_Skeleton-based_Video_Anomaly_Detection_ICCV_2023_paper.html):Multimodal motion conditioned diffusion model for skeleton-based video anomaly detection, 📰 `ICCV`  [code](https://github.com/aleflabo/MoCoDAD)

### 1.4 Model Refinement

#### 1.4.1 Pseudo Anomalies

🗓️ **2021**

- 📄 [LNRA](https://arxiv.org/abs/2110.09742):Learning not to reconstruct anomalies, 📰 `BMVC` [code](https://github.com/aseuteurideu/LearningNotToReconstructAnomalies)

- 📄 [G2D](https://openaccess.thecvf.com/content/WACV2021/html/Pourreza_G2D_Generate_to_Detect_Anomaly_WACV_2021_paper.html):G2d: Generate to detect anomaly, 📰 `WACV`  [code](https://github.com/masoudpz/G2D_generate_to_detect_anomaly)

- 📄 [BAF](https://ieeexplore.ieee.org/abstract/document/9410375):A background-agnostic framework with adversarial training for abnormal event detection in video, 📰 `TPAMI` [code](https://github.com/lilygeorgescu/AED)

🗓️ **2022**

- 📄 [OGNet+](https://ieeexplore.ieee.org/abstract/document/9887825):Stabilizing adversarially learned one-class novelty detection using pseudo anomalies, 📰 `TIP`

- 📄 [MBPA](https://ieeexplore.ieee.org/abstract/document/9826251):Limiting reconstruction capability of autoencoders using moving backward pseudo anomalies, 📰 `UR`

🗓️ **2023**

- 📄 [DSS-NET](https://ieeexplore.ieee.org/abstract/document/10174739):Dss-net: Dynamic self-supervised network for video anomaly detection, 📰 `TMM` 

- 📄 [PseudoBound](https://www.sciencedirect.com/science/article/pii/S092523122300228X):Pseudobound: Limiting the anomaly reconstruction capability of one-class classifiers using pseudo anomalies, 📰 `Neurocomputing`

- 📄 [PFMF](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Generating_Anomalies_for_Video_Anomaly_Detection_With_Prompt-Based_Feature_Mapping_CVPR_2023_paper.html):Generating anomalies for video anomaly detection with prompt-based feature mapping, 📰 `CVPR`

#### 1.4.2 Memory Bank

🗓️ **2019**

- 📄 [MemAE](https://openaccess.thecvf.com/content_ICCV_2019/html/Gong_Memorizing_Normality_to_Detect_Anomaly_Memory-Augmented_Deep_Autoencoder_for_Unsupervised_ICCV_2019_paper.html): Memorizing normality to detect anomaly: Memory-augmented deep autoencoder for unsupervised anomaly detection, 📰 `ICCV` [code](https://github.com/donggong1/memae-anomaly-detection)

🗓️ **2020**

- 📄 [MNAD](https://openaccess.thecvf.com/content_CVPR_2020/html/Park_Learning_Memory-Guided_Normality_for_Anomaly_Detection_CVPR_2020_paper.html\):Learning memory-guided normality for anomaly detection, 📰 `CVPR`  [code](https://github.com/cvlab-yonsei/MNAD)  [homepage](https://cvlab.yonsei.ac.kr/projects/MNAD/)

🗓️ **2021**

- 📄 [MPN](https://openaccess.thecvf.com/content/CVPR2021/html/Lv_Learning_Normal_Dynamics_in_Videos_With_Meta_Prototype_Network_CVPR_2021_paper.html):Learning normal dynamics in videos with meta prototype network, 📰 `CVPR`  [code](https://github.com/ktr-hubrt/MPN)

🗓️ **2022**

- 📄 [EPAP-Net](https://dl.acm.org/doi/abs/10.1145/3503161.3548000):Anomaly warning: Learning and memorizing future semantic patterns for unsupervised ex-ante potential anomaly prediction, 📰 `ACM MM`

- 📄 [CAFE](https://dl.acm.org/doi/abs/10.1145/3503161.3547944):Effective video abnormal event detection by learning a consistency-aware high-level feature extractor, 📰 `ACM MM`

- 📄 [DLAN-AC](https://link.springer.com/chapter/10.1007/978-3-031-19772-7_24):Dynamic local aggregation network with adaptive clusterer for anomaly detection, 📰 `ECCV`  [code](https://github.com/Beyond-Zw/DLAN-AC)

🗓️ **2023**

- 📄 [DMAD](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Diversity-Measurable_Anomaly_Detection_CVPR_2023_paper.html):Diversity-measurable anomaly detection, 📰 `CVPR` [code](https://github.com/FlappyPeggy/DMAD)

- 📄 [SVN](https://www.sciencedirect.com/science/article/abs/pii/S0950705123007360):Stochastic video normality network for abnormal event detection in surveillance videos, 📰 `KBS`

- 📄 [LERF](https://ojs.aaai.org/index.php/AAAI/article/view/25334):Learning event-relevant factors for video anomaly detection, 📰 `AAAI`

- 📄 [MAAM-Net](https://www.sciencedirect.com/science/article/abs/pii/S0031320323000365):Memory-augmented appearance-motion network for video anomaly detection, 📰 `PR`

🗓️ **2024**

- 📄 [STU-Net](https://ieeexplore.ieee.org/abstract/document/10462921):Context recovery and knowledge retrieval: A novel two-stream framework for video anomaly detection, 📰 `TIP`  [homepage](https://github.com/zugexiaodui/TwoStreamUVAD)

### 1.5 Model Output

#### 1.5.1 Frame Level

#### 1.5.2 Pixel Level

🗓️ **2022**

- 📄 [UPformer](https://dl.acm.org/doi/abs/10.1145/3503161.3548082):Pixel-level anomaly detection via uncertainty-aware prototypical transformer, 📰 `ACM MM`

## 2. Weakly Supervised Video Anomaly Detection

🗓️ **2018**

- 📄 [DeepMIL](https://openaccess.thecvf.com/content_cvpr_2018/html/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.html): Real-world anomaly detectionin surveillance videos, 📰 `CVPR` [code](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018)[homepage](http://crcv.ucf.edu/projects/real-world/)

### 2.1 Model Input

#### 2.1.1 RGB

🗓️ **2018**

- 📄 [DeepMIL](https://openaccess.thecvf.com/content_cvpr_2018/html/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.html): Real-world anomaly detectionin surveillance videos, 📰 `CVPR` [code1](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018) [code2](https://github.com/Roc-Ng/DeepMIL)  [homepage](http://crcv.ucf.edu/projects/real-world/)

🗓️ **2019**

- 📄 [GCN](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhong_Graph_Convolutional_Label_Noise_Cleaner_Train_a_Plug-And-Play_Action_Classifier_CVPR_2019_paper.html):Graph convolutional label noise cleaner: Train a plug-and-play action classifier for anomaly detection, 📰 `CVPR`  [code](https://github.com/jx-zhong-for-academic-purpose/GCN-Anomaly-Detection)

🗓️ **2020**

- 📄 [CLAWS](https://link.springer.com/chapter/10.1007/978-3-030-58542-6_22): Claws: Clustering assisted weakly supervised learning with normalcy suppression for anomalous event detection, 📰 `ECCV`  [code](https://github.com/xaggi/claws_eccv)

- 📄 [HLNet](https://link.springer.com/chapter/10.1007/978-3-030-58577-8_20):Not only look, but also listen: Learning multimodal violence detection under weak supervision, 📰 `ECCV`  [code](https://github.com/Roc-Ng/XDVioDet) [homepage](https://roc-ng.github.io/XD-Violence/)

🗓️ **2022**

- 📄 [S3R](https://link.springer.com/chapter/10.1007/978-3-031-19778-9_42):Self-supervised sparse representation for video anomaly detection, 📰 `ECCV`  [code](https://github.com/louisYen/S3R)

- 📄 [GCN+](https://www.sciencedirect.com/science/article/abs/pii/S0925231222000443):Weakly-supervised anomaly detection in video surveillance via graph convolutional label noise cleaning, 📰 `Neurocomputing`

- 📄 [MSL](https://ojs.aaai.org/index.php/AAAI/article/view/20028):Self-training multi-sequence learning with transformer for weakly supervised video anomaly detection, 📰 `AAAI`

🗓️ **2023**

- 📄 [BN-WVAD](https://arxiv.org/abs/2311.15367):Batchnorm-based weakly supervised video anomaly detection, 📰 `Arxiv`  [code](https://github.com/cool-xuan/BN-WVAD)

- 📄 [LSTC](https://ieeexplore.ieee.org/abstract/document/10219868):Long-short temporal co-teaching for weakly supervised video anomaly detection, 📰 `ICME`  [code](https://github.com/shengyangsun/LSTC_VAD)

🗓️ **2024**

- 📄 [AlMarri Salem et al.](https://openaccess.thecvf.com/content/WACV2024W/RWS/html/AlMarri_A_Multi-Head_Approach_With_Shuffled_Segments_for_Weakly-Supervised_Video_Anomaly_WACVW_2024_paper.html): A multi-head approach with shuffled segments for weakly-supervised video anomaly detection, 📰 `WACV`

- 📄 [OVVAD](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Open-Vocabulary_Video_Anomaly_Detection_CVPR_2024_paper.html):Open-vocabulary video anomaly detection, 📰 `CVPR`

#### 2.1.2 Optical Flow

🗓️ **2019**

- 📄 [GCN](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhong_Graph_Convolutional_Label_Noise_Cleaner_Train_a_Plug-And-Play_Action_Classifier_CVPR_2019_paper.html):Graph convolutional label noise cleaner: Train a plug-and-play action classifier for anomaly detection, 📰 `CVPR`  [code](https://github.com/jx-zhong-for-academic-purpose/GCN-Anomaly-Detection)

🗓️ **2020**

- 📄 [AR-NET](https://ieeexplore.ieee.org/abstract/document/9102722):Weakly supervised video anomaly detection via center-guided discriminative learning, 📰 `ICME`  [code](https://github.com/wanboyang/Anomaly_AR_Net_ICME_2020)

#### 2.1.3 Audio

🗓️ **2021**

- 📄 [FVAL](https://ieeexplore.ieee.org/abstract/document/9413686):Violence detection in videos based on fusing visual and audio information, 📰 `ICASSP`

🗓️ **2023**

- 📄 [HyperVD](https://arxiv.org/abs/2305.18797):Learning weakly supervised audio-visual violence detection in hyperbolic space, 📰 `Arxiv`  [code](https://github.com/xiaogangpeng/HyperVD)

#### 2.1.4 Text

🗓️ **2023**

- 📄 [PEL4VAD](https://arxiv.org/abs/2306.14451):Learning prompt-enhanced context features for weakly-supervised video anomaly detection, 📰 `Arxiv`  [code](https://github.com/yujiangpu20/PEL4VAD)

- 📄 [TEVAD](https://openaccess.thecvf.com/content/CVPR2023W/O-DRUM/html/Chen_TEVAD_Improved_Video_Anomaly_Detection_With_Captions_CVPRW_2023_paper.html):Tevad: Improved video anomaly detection with captions, 📰 `CVPRW` [code](https://github.com/coranholmes/TEVAD)

🗓️ **2024**

- 📄 [LAP](https://arxiv.org/abs/2403.01169):Learn suspected anomalies from event prompts for video anomaly detection, 📰 `Arxiv`

- 📄 [ALAN](https://ieeexplore.ieee.org/abstract/document/10471334):Toward video anomaly retrieval from video anomaly detection: New benchmarks and model, 📰 `TIP`

#### 2.1.5 Hybrid

🗓️ **2020**

- 📄 [AR-NET](https://ieeexplore.ieee.org/abstract/document/9102722):Weakly supervised video anomaly detection via center-guided discriminative learning, 📰 `ICME`  [code](https://github.com/wanboyang/Anomaly_AR_Net_ICME_2020)

🗓️ **2022**

- 📄 [ACF_MMVD](https://ieeexplore.ieee.org/abstract/document/9746422):Look, listen and pay more attention: Fusing multi-modal information for video violence detection, 📰 `ICASSP`  [code]( https://github.com/DL-Wei/ACF_MMVD)

- 📄 [MSFA](https://ieeexplore.ieee.org/abstract/document/9926192):Msaf: Multimodal supervise-attention enhanced fusion for video anomaly detection, 📰 `SPL`  [homepage](https://github.com/Video-AD/MSFA)

- 📄 [MACIL_SD](https://dl.acm.org/doi/abs/10.1145/3503161.3547868):Modality-aware contrastive instance learning with self-distillation for weakly-supervised audio-visual violence detection, 📰 `ACM MM`  [code](https://github.com/JustinYuu/MACIL_SD)

- 📄 [HL-Net+](https://ieeexplore.ieee.org/abstract/document/9699377):Weakly supervised audio-visual violence detection, 📰 `TMM`

🗓️ **2024**

- 📄 [UCA](https://openaccess.thecvf.com/content/CVPR2024/html/Yuan_Towards_Surveillance_Video-and-Language_Understanding_New_Dataset_Baselines_and_Challenges_CVPR_2024_paper.html):Towards surveillance video-and-language understanding: New dataset baselines and challenges, 📰 `CVPR`  [homepage](https://github.com/Xuange923/Surveillance-Video-Understanding)

### 2.2 Methodology

#### 2.2.1 One-Stage MIL

🗓️ **2018**

- 📄 [DeepMIL](https://openaccess.thecvf.com/content_cvpr_2018/html/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.html): Real-world anomaly detectionin surveillance videos, 📰 `CVPR` [code1](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018) [code2](https://github.com/Roc-Ng/DeepMIL) ([homepage](http://crcv.ucf.edu/projects/real-world/)

🗓️ **2019**

- 📄 [MAF](https://arxiv.org/abs/1907.10211):Motion-aware feature for improved video anomaly detection 📰 `BMVC`

- 📄 [TCN-IBL](https://ieeexplore.ieee.org/abstract/document/8803657):Temporal convolutional network with complementary inner bag loss for weakly supervised anomaly detection, 📰 `ICIP`

🗓️ **2020**

- 📄 [HLNet](https://link.springer.com/chapter/10.1007/978-3-030-58577-8_20):Not only look, but also listen: Learning multimodal violence detection under weak supervision, 📰 `ECCV`  [code](https://github.com/Roc-Ng/XDVioDet)

🗓️ **2022**

- 📄 [CNL](https://ieeexplore.ieee.org/abstract/document/9739763):Collaborative normality learning framework for weakly supervised video anomaly detection, 📰 ` TCAS-II`

#### 2.2.2 Two-Stage Self-Training

🗓️ **2019**

- 📄 [GCN](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhong_Graph_Convolutional_Label_Noise_Cleaner_Train_a_Plug-And-Play_Action_Classifier_CVPR_2019_paper.html):Graph convolutional label noise cleaner: Train a plug-and-play action classifier for anomaly detection, 📰 `CVPR`

🗓️ **2021**

- 📄 [MIST](https://openaccess.thecvf.com/content/CVPR2021/html/Feng_MIST_Multiple_Instance_Self-Training_Framework_for_Video_Anomaly_Detection_CVPR_2021_paper.html):Mist: Multiple instance self-training framework for video anomaly detection, 📰 `CVPR`  [code](https://github.com/fjchange/MIST_VAD)  [homepage](https://kiwi-fung.win/2021/04/28/MIST/)

🗓️ **2022**

- 📄 [MSL](https://ojs.aaai.org/index.php/AAAI/article/view/20028):Self-training multi-sequence learning with transformer for weakly supervised video anomaly detection, 📰 `AAAI`

🗓️ **2023**

- 📄 [CUPL](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Exploiting_Completeness_and_Uncertainty_of_Pseudo_Labels_for_Weakly_Supervised_CVPR_2023_paper.html):Exploiting completeness and uncertainty of pseudo labels for weakly supervised video anomaly detection, 📰 `CVPR`  [code](https://github.com/ArielZc/CU-Net)

🗓️ **2024**

- 📄 [TPWNG](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Text_Prompt_with_Normality_Guidance_for_Weakly_Supervised_Video_Anomaly_CVPR_2024_paper.html):Text prompt with normality guidance for weakly supervised video anomaly detection, 📰 `CVPR`

### 2.3 Refinement Strategy

#### 2.3.1 Temporal Modeling

🗓️ **2020**

- 📄 [HLNet](https://link.springer.com/chapter/10.1007/978-3-030-58577-8_20):Not only look, but also listen: Learning multimodal violence detection under weak supervision, 📰 `ECCV`  [code](https://github.com/Roc-Ng/XDVioDet)

🗓️ **2021**

- 📄 [CTR](https://ieeexplore.ieee.org/abstract/document/9369126):Learning causal temporal relation and feature discrimination for anomaly detection, 📰 `TIP`

- 📄 [RTFM](https://openaccess.thecvf.com/content/ICCV2021/html/Tian_Weakly-Supervised_Video_Anomaly_Detection_With_Robust_Temporal_Feature_Magnitude_Learning_ICCV_2021_paper.html):Weakly-supervised video anomaly detection with robust temporal feature magnitude learning, 📰 `ICCV`  [code](https://github.com/tianyu0207/RTFM)

- 📄 [CA-Net](https://ieeexplore.ieee.org/abstract/document/9540293):Contrastive attention for video anomaly detection, 📰 `TMM`  [code](https://github.com/changsn/Contrastive-Attention-for-Video-Anomaly-Detection)

- 📄 [CRF](https://openaccess.thecvf.com/content/ICCV2021/html/Purwanto_Dance_With_Self-Attention_A_New_Look_of_Conditional_Random_Fields_ICCV_2021_paper.html):Dance with self-attention: A new look of conditional random fields on anomaly detection in videos, 📰 `ICCV`

🗓️ **2022**

- 📄 [MSL](https://ojs.aaai.org/index.php/AAAI/article/view/20028):Self-training multi-sequence learning with transformer for weakly supervised video anomaly detection, 📰 `AAAI`

- 📄 [DAR](https://ieeexplore.ieee.org/abstract/document/9926133):Decouple and resolve: transformer-based models for online anomaly detection from weakly labeled videos, 📰 `TIFS`

- 📄 [WAGCN](https://ieeexplore.ieee.org/abstract/document/9968312):Adaptive graph convolutional networks for weakly supervised anomaly detection in videos, 📰 `SPL`

- 📄 [SGTDT](https://ieeexplore.ieee.org/abstract/document/10002867):Weakly supervised video anomaly detection via self-guided temporal discriminative transformer, 📰 `TCYB`

- 📄 [MLAD](https://ieeexplore.ieee.org/abstract/document/9774903):Weakly supervised anomaly detection in videos considering the openness of events, 📰 `TITS`

🗓️ **2023**

- 📄 [CMRL](https://openaccess.thecvf.com/content/CVPR2023/html/Cho_Look_Around_for_Anomalies_Weakly-Supervised_Anomaly_Detection_via_Context-Motion_Relational_CVPR_2023_paper.html): Look around for anomalies: weakly-supervised anomaly detection via context-motion relational learning, 📰 `CVPR`

- 📄 [CBCG](https://ieeexplore.ieee.org/abstract/document/10219658):Weakly supervised video anomaly detection based on cross-batch clustering guidance, 📰 `ICME`

- 📄 [DMU](https://ojs.aaai.org/index.php/AAAI/article/view/25489):Dual memory units with uncertainty regulation for weakly supervised video anomaly detection, 📰 `AAAI`  [code](https://github.com/henrryzh1/UR-DMU)

#### 2.3.2 Spatio-Temporal Modeling

🗓️ **2022**

- 📄 [STA-Net](https://ieeexplore.ieee.org/abstract/document/9746822):Learning task-specific representation for video anomaly detection with spatialtemporal attention, 📰 `ICASSP`

- 📄 [SSRL](https://link.springer.com/chapter/10.1007/978-3-031-19772-7_20):Scale-aware spatio-temporal relation learning for video anomaly detection, 📰 `ECCV`

🗓️ **2023**

- 📄 [LSTC](https://ieeexplore.ieee.org/abstract/document/10219868):Long-short temporal co-teaching for weakly supervised video anomaly detection, 📰 `ICME`  [code](https://github.com/shengyangsun/LSTC_VAD)

🗓️ **2024**

- 📄 [MSIP](https://ieeexplore.ieee.org/abstract/document/10447603): Learning spatio-temporal relations with multi-scale integrated perception for video anomaly detection, 📰 `ICASSP`

#### 2.3.3 MIL-Based Refinement

🗓️ **2019**

- 📄 [Social-MIL](https://ieeexplore.ieee.org/abstract/document/8909882):Social mil: Interaction-aware for crowd anomaly detection, 📰 `AVSS`

🗓️ **2022**

- 📄 [MCR](https://ieeexplore.ieee.org/abstract/document/9860012):Multiscale continuity-aware refinement network for weakly supervised video anomaly detection, 📰 `ICME`

- 📄 [BN-SVP](https://openaccess.thecvf.com/content/CVPR2022/html/Sapkota_Bayesian_Nonparametric_Submodular_Video_Partition_for_Robust_Anomaly_Detection_CVPR_2022_paper.html):Bayesian nonparametric submodular video partition for robust anomaly detection, 📰 `CVPR`  [code](https://github.com/ritmininglab/BN-SVP)

🗓️ **2023**

- 📄 [NGMIL](https://openaccess.thecvf.com/content/WACV2023/html/Park_Normality_Guided_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_WACV_2023_paper.html):Normality guided multiple instance learning for weakly supervised video anomaly detection, 📰 `WACV`

- 📄 [UMIL](https://openaccess.thecvf.com/content/CVPR2023/html/Lv_Unbiased_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_Detection_CVPR_2023_paper.html):Unbiased multiple instance learning for weakly supervised video anomaly detection, 📰 `CVPR`  [code]( https://github.com/ktr-hubrt/UMIL)

- 📄 [MGFN](https://ojs.aaai.org/index.php/AAAI/article/view/25112):Mgfn: Magnitude-contrastive glance-and-focus network for weakly-supervised video anomaly detection, 📰 `AAAI`  [code](https://github.com/carolchenyx/MGFN)

🗓️ **2024**

- 📄 [LAP](https://arxiv.org/abs/2403.01169):Learn suspected anomalies from event prompts for video anomaly detection, 📰 `Arxiv`

- 📄 [PE-MIL](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Prompt-Enhanced_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_Detection_CVPR_2024_paper.html): Prompt-enhanced multiple instance learning for weakly supervised video anomaly detection, 📰 `CVPR`

#### 2.3.4 Feature Metric Learning

🗓️ **2019**

- 📄 [TCN-IBL](https://ieeexplore.ieee.org/abstract/document/8803657):Temporal convolutional network with complementary inner bag loss for weakly supervised anomaly detection, 📰 `ICIP`

🗓️ **2021**

- 📄 [CTR](https://ieeexplore.ieee.org/abstract/document/9369126):Learning causal temporal relation and feature discrimination for anomaly detection, 📰 `TIP`

🗓️ **2022**

- 📄 [SGTDT](https://ieeexplore.ieee.org/abstract/document/10002867):Weakly supervised video anomaly detection via self-guided temporal discriminative transformer, 📰 `TCYB`

🗓️ **2023**

- 📄 [BN-WVAD](https://arxiv.org/abs/2311.15367):Batchnorm-based weakly supervised video anomaly detection, 📰 `Arxiv`  [code](https://github.com/cool-xuan/BN-WVAD)

- 📄 [PEL4VAD](https://arxiv.org/abs/2306.14451):Learning prompt-enhanced context features for weakly-supervised video anomaly detection, 📰 `Arxiv`  [code](https://github.com/yujiangpu20/PEL4VAD)

- 📄 [TeD-SPAD](https://openaccess.thecvf.com/content/ICCV2023/html/Fioresi_TeD-SPAD_Temporal_Distinctiveness_for_Self-Supervised_Privacy-Preservation_for_Video_Anomaly_Detection_ICCV_2023_paper.html):Ted-spad: Temporal distinctiveness for self-supervised privacy-preservation for video anomaly detection, 📰 `ICCV`  [code](https://github.com/UCF-CRCV/TeD-SPAD)

- 📄 [CLAWS+](https://ieeexplore.ieee.org/abstract/document/10136845):Clustering aided weakly supervised training to detect anomalous events in surveillance videos, 📰 `TNNLS`

🗓️ **2024**

- 📄 [LAP](https://arxiv.org/abs/2403.01169):Learn suspected anomalies from event prompts for video anomaly detection, 📰 `Arxiv`

#### 2.3.5 Knowledge Distillation

🗓️ **2022**

- 📄 [MACIL-SD](https://dl.acm.org/doi/abs/10.1145/3503161.3547868):Modality-aware contrastive instance learning with self-distillation for weakly-supervised audio-visual violence detection, 📰 `ACM MM`  [code](https://github.com/JustinYuu/MACIL_SD)

🗓️ **2023**

- 📄 [DPK](https://ieeexplore.ieee.org/abstract/document/10136845):Distilling privileged knowledge for anomalous event detection from weakly labeled videos, 📰 `TNNLS`

#### 2.3.6 Leveraging Large Models:

🗓️ **2023**

- 📄 [TEVAD](https://openaccess.thecvf.com/content/CVPR2023W/O-DRUM/html/Chen_TEVAD_Improved_Video_Anomaly_Detection_With_Captions_CVPRW_2023_paper.html):Tevad: Improved video anomaly detection with captions, 📰 `CVPRW`

- 📄 [CLIP-TSA](https://ieeexplore.ieee.org/abstract/document/10222289):Clip-tsa: Clip-assisted temporal self-attention for weakly-supervised video anomaly detection, 📰 `ICIP`  [code](https://github.com/joos2010kj/CLIP-TSA)

🗓️ **2024**

- 📄 [UCA](https://openaccess.thecvf.com/content/CVPR2024/html/Yuan_Towards_Surveillance_Video-and-Language_Understanding_New_Dataset_Baselines_and_Challenges_CVPR_2024_paper.html):Towards surveillance video-and-language understanding: New dataset baselines and challenges, 📰 `CVPR` [homepage](https://xuange923.github.io/Surveillance-Video-Understanding)

- 📄 [VadCLIP](https://ojs.aaai.org/index.php/AAAI/article/view/28423):Vadclip: Adapting vision-language models for weakly supervised video anomaly detection, 📰 `AAAI`  [code]( https://github.com/nwpu-zxr/VadCLIP)

- 📄 [Holmes-VAD](https://arxiv.org/abs/2406.12235):Holmes-vad: Towards unbiased and explainable video anomaly detection via multi-modal llm, 📰 `Arxiv` [code](https://github.com/pipixin321/HolmesVAD) [homepage](https://holmesvad.github.io/)

- 📄 [VADor w LSTC](https://arxiv.org/abs/2401.05702):Video anomaly detection and explanation via large language models, 📰 `Arxiv`

- 📄 [LAVAD](https://openaccess.thecvf.com/content/CVPR2024/html/Zanella_Harnessing_Large_Language_Models_for_Training-free_Video_Anomaly_Detection_CVPR_2024_paper.html): Harnessing large language models for training-free video anomaly detection, 📰 `CVPR` [code](https://github.com/lucazanella/lavad)  [homepage](https://lucazanella.github.io/lavad/)

- 📄 [STPrompt](https://arxiv.org/abs/2408.05905):Weakly supervised video anomaly detection and localization with spatio-temporal prompts, 📰 `ACM MM`

### 2.4 Model Output

#### 2.4.1 Frame Level

#### 2.4.2 Pixel Level

🗓️ **2019**

- 📄 [Background-bias](https://dl.acm.org/doi/abs/10.1145/3343031.3350998):Exploring background-bias for anomaly detection in surveillance videos, 📰 `ACM MM` [code](https://github.com/xuzero/UCFCrime_BoundingBox_Annotation)

🗓️ **2021**

- 📄 [WSSTAD](https://arxiv.org/abs/2108.03825):Weakly-supervised spatio-temporal anomaly detection in surveillance video, 📰 `IJCAI`

## 3. Fully Supervised Video Anomaly Detection

### 3.1 Appearance Input

🗓️ **2016**

- 📄 [TS-LSTM](https://link.springer.com/chapter/10.1007/978-981-10-3002-4_43):Multi-stream deep networks for person to person violence detection in videos, 📰 `CCPR`

🗓️ **2017**

- 📄 [FightNet](https://iopscience.iop.org/article/10.1088/1742-6596/844/1/012044/meta):Violent interaction detection in video based on deep learning, 📰 `JPCS`

🗓️ **2019**

- 📄 [Sub-Vio](https://ieeexplore.ieee.org/abstract/document/8682833):Toward subjective violence detection in videos, 📰 `ICASSP`

- 📄 [CCTV-Fights](https://ieeexplore.ieee.org/abstract/document/8683676):Detection of real-world fights in surveillance videos, 📰 `ICASSP` [homepage](http://rose1.ntu.edu.sg/Datasets/cctvFights.asp)

### 3.2 Motion Input

🗓️ **2016**

- 📄 [TS-LSTM](https://link.springer.com/chapter/10.1007/978-981-10-3002-4_43):Multi-stream deep networks for person to person violence detection in videos, 📰 `CCPR`

🗓️ **2017**

- 📄 [ConvLSTM](https://ieeexplore.ieee.org/abstract/document/8078468):Learning to detect violent videos using convolutional long short-term memory, 📰 `AVSS`  [code](https://github.com/swathikirans/violence-recognition-pytorch)

🗓️ **2018**

- 📄 [BiConvLSTM](https://openaccess.thecvf.com/content_eccv_2018_workshops/w10/html/Hanson_Bidirectional_Convolutional_LSTM_for_the_Detection_of_Violence_in_Videos_ECCVW_2018_paper.html):Bidirectional convolutional lstm for the detection of violence in videos, 📰 `ECCVW`

🗓️ **2020**

- 📄 [MM-VD](https://ieeexplore.ieee.org/abstract/document/9054018):Multimodal violence detection in videos, 📰 `ICASSP`

### 3.3 Skeleton Input

 🗓️ **2018**

- 📄 [DSS](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8575376):Eye in the sky: Real-time drone surveillance system for violent individuals identification using scatternet hybrid deep learning network, 📰 `CVPRW`
  
  🗓️ **2020**

- 📄 [SPIL](https://link.springer.com/chapter/10.1007/978-3-030-58548-8_5):Human interaction learning on 3d skeleton point clouds for video violence recognition, 📰 `ECCV`

### 3.4 Audio Input

 🗓️ **2020**

- 📄 [MM-VD](https://ieeexplore.ieee.org/abstract/document/9054018):Multimodal violence detection in videos, 📰 `ICASSP`

### 3.5 Hybrid Input

 🗓️ **2021**

- 📄 [FlowGatedNet](https://ieeexplore.ieee.org/abstract/document/9412502):Rwf-2000: an open large scale video database for violence detection, 📰 `ICPR`  [code](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection)

🗓️ **2022**

- 📄 [MutualDis](https://link.springer.com/chapter/10.1007/978-3-031-18913-5_48):Multimodal violent video recognition based on mutual distillation, 📰 `PRCV`

🗓️ **2023**

- 📄 [HSCD](https://www.sciencedirect.com/science/article/pii/S1077314223001194): Human skeletons and change detection for efficient violence detection in surveillance videos, 📰 `CVIU`  [code](https://github.com/atmguille/Violence-Detection-With-Human-Skeletons)

## 4. Unsupervised Video Anomaly Detection

### 4.1 Pseudo Label Based Paradigm

 🗓️ **2018**

- 📄 [DAW](https://dl.acm.org/doi/abs/10.1145/3240508.3240615):Detecting abnormality without knowing normality: A two-stage approach for unsupervised video abnormal event detection, 📰 `ACM MM`

🗓️ **2020**

- 📄 [STDOR](https://openaccess.thecvf.com/content_CVPR_2020/html/Pang_Self-Trained_Deep_Ordinal_Regression_for_End-to-End_Video_Anomaly_Detection_CVPR_2020_paper.html):Self-trained deep ordinal regression for end-to-end video anomaly detection, 📰 `CVPR`

🗓️ **2022**

- 📄 [GCL](https://openaccess.thecvf.com/content/CVPR2022/html/Zaheer_Generative_Cooperative_Learning_for_Unsupervised_Video_Anomaly_Detection_CVPR_2022_paper.html):Generative cooperative learning for unsupervised video anomaly detection, 📰 `CVPR`

🗓️ **2024**

- 📄 [C2FPL](https://openaccess.thecvf.com/content/WACV2024/html/Al-lahham_A_Coarse-To-Fine_Pseudo-Labeling_C2FPL_Framework_for_Unsupervised_Video_Anomaly_Detection_WACV_2024_paper.html):A coarse-to-fine pseudo-labeling (c2fpl) framework for unsupervised video anomaly detection, 📰 `WACV`  [code](https://github.com/AnasEmad11/C2FPL)

### 4.2 Change Detection Based Paradigm

🗓️ **2016**

- 📄 [ADF](https://link.springer.com/chapter/10.1007/978-3-319-46454-1_21):A discriminative framework for anomaly detection in large videos, 📰 `ECCV`  [code](https://github.com/alliedel/anomalyframework_ECCV2016)

🗓️ **2017**

- 📄 [Unmasking](https://openaccess.thecvf.com/content_iccv_2017/html/Ionescu_Unmasking_the_Abnormal_ICCV_2017_paper.html):Unmasking the abnormal events in video, 📰 `ICCV`

🗓️ **2018**

- 📄 [MC2ST](https://chunliangli.github.io/docs/18bmvcAnomaly.pdf):Classifier two sample test for video anomaly detections, 📰 `BMVC`  [code](https://github.com/MYusha/Video-Anomaly-Detection)

🗓️ **2022**

- 📄 [TMAE](https://ieeexplore.ieee.org/abstract/document/9859873):Detecting anomalous events from unlabeled videos via temporal masked autoencoding, 📰 `ICME`

### 4.3 Others

🗓️ **2021**

- 📄 [DUAD](https://openaccess.thecvf.com/content/WACV2021/html/Li_Deep_Unsupervised_Anomaly_Detection_WACV_2021_paper.html):Deep unsupervised anomaly detection, 📰 `WACV`

🗓️ **2022**

- 📄 [CIL](https://ojs.aaai.org/index.php/AAAI/article/view/20053):A causal inference look at unsupervised video anomaly detection, 📰 `AAAI`

- 📄 [LBR-SPR](https://openaccess.thecv.com/content/CVPR2022/html/Yu_Deep_Anomaly_Discovery_From_Unlabeled_Videos_via_Normality_Advantage_and_CVPR_2022_paper.html):Deep anomaly discovery from unlabeled videos via normality advantage and self-paced refinement, 📰 `CVPR`  [code](https://github.com/yuguangnudt/LBR_SPR)

## 5. Open Set Supervised Video Anomaly Detection

### 5.1 Open-Set VAD

🗓️ **2019**

- 📄 [MLEP](https://www.ijcai.org/proceedings/2019/0419.pdf):Margin learning embedded prediction for video anomaly detection with a few anomalies, 📰 `IJCAI`  [code]( https://github.com/svip-lab/MLEP)

🗓️ **2022**

- 📄 [UBnormal](https://openaccess.thecvf.com/content/CVPR2022/html/Acsintoae_UBnormal_New_Benchmark_for_Supervised_Open-Set_Video_Anomaly_Detection_CVPR_2022_paper.html):Ubnormal: New benchmark for supervised open-set video anomaly detection, 📰 `CVPR`  [code](https://github.com/lilygeorgescu/UBnormal)

- 📄 [OSVAD](https://link.springer.com/chapter/10.1007/978-3-031-19830-4_23):Towards open set video anomaly detection, 📰 `ECCV`

🗓️ **2024**

- 📄 [OVVAD](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Open-Vocabulary_Video_Anomaly_Detection_CVPR_2024_paper.html):Open-vocabulary video anomaly detection, 📰 `CVPR`

### 5.2 Few-Shot VAD

🗓️ **2020**

- 📄 [FSSA](https://link.springer.com/chapter/10.1007/978-3-030-58558-7_8):Few-shot scene-adaptive anomaly detection, 📰 `ECCV`  [code](https://github.com/yiweilu3/Few-shot-Scene-adaptive-Anomaly-Detection)

🗓️ **2021**

- 📄 [AADNet](https://link.springer.com/chapter/10.1007/978-3-030-88007-1_26):Adaptive anomaly detection network for unseen scene without fine-tuning, 📰 `PRCV`

🗓️ **2022**

- 📄 [VADNet](https://ieeexplore.ieee.org/abstract/document/9976040):Boosting variational inference with margin learning for few-shot scene-adaptive anomaly detection, 📰 `TCSVT`  [code](https://github.com/huangxx156/VADNet)

🗓️ **2023**

- 📄 [zxVAD](https://openaccess.thecvf.com/content/WACV2023/html/Aich_Cross-Domain_Video_Anomaly_Detection_Without_Target_Domain_Adaptation_WACV_2023_paper.html):Cross-domain video anomaly detection without target domain adaptation, 📰 `WACV`

## Performance Comparison

The following tables are the performance comparison of semi-supervised VAD, weakly supervised VAD, fully supervised VAD, and unsupervised VAD methods as reported in the literature. For semi-supervised, weakly supervised, and unsupervised VAD methods, the evaluation metric used is AUC (%) and AP ( XD-Violence, %), while for  fully supervised VAD methods, the metric is Accuracy (%).

- **Quantitative Performance Comparison of Semi-supervised Methods on Public Datasets.**

| Method     | Publication | Methodology            | Ped1 | Ped2 | Avenue | ShanghaiTech | UBnormal |
| ---------- | -----------:| ---------------------- |:----:|:----:|:------:|:------------:|:--------:|
| AMDN       | BMVC 2015   | One-class classifier   | 92.1 | 90.8 | -      | -            | -        |
| ConvAE     | CVPR 2016   | Reconstruction         | 81.0 | 90.0 | 72.0   | -            | -        |
| STAE       | ACMMM 2017  | Hybrid                 | 92.3 | 91.2 | 80.9   | -            | -        |
| StackRNN   | ICCV 2017   | Sparse coding          | -    | 92.2 | 81.7   | 68.0         | -        |
| FuturePred | CVPR 2018   | Prediction             | 83.1 | 95.4 | 85.1   | 72.8         | -        |
| DeepOC     | TNNLS 2019  | One-class classifier   | 83.5 | 96.9 | 86.6   | -            | -        |
| MemAE      | ICCV 2019   | Reconstruction         | -    | 94.1 | 83.3   | 71.2         | -        |
| AnoPCN     | ACMMM 2019  | Prediction             | -    | 96.8 | 86.2   | 73.6         | -        |
| ObjectAE   | CVPR 2019   | One-class classifier   | -    | 97.8 | 90.4   | 84.9         | -        |
| BMAN       | TIP 2019    | Prediction             | -    | 96.6 | 90.0   | 76.2         | -        |
| sRNN-AE    | TPAMI 2019  | Sparse coding          | -    | 92.2 | 83.5   | 69.6         | -        |
| ClusterAE  | ECCV 2020   | Reconstruction         | -    | 96.5 | 86.0   | 73.3         | -        |
| MNAD       | CVPR 2020   | Reconstruction         | -    | 97.0 | 88.5   | 70.5         | -        |
| VEC        | ACMMM 2020  | Cloze test             | -    | 97.3 | 90.2   | 74.8         | -        |
| AMMC-Net   | AAAI 2021   | Prediction             | -    | 96.6 | 86.6   | 73.7         | -        |
| MPN        | CVPR 2021   | Prediction             | 85.1 | 96.9 | 89.5   | 73.8         | -        |
| HF$^2$-VAD | ICCV 2021   | Hybrid                 | -    | 99.3 | 91.1   | 76.2         | -        |
| BAF        | TPAMI 2021  | One-class classifier   |      | 98.7 | 92.3   | 82.7         | 59.3     |
| Multitask  | CVPR 2021   | Multiple tasks         | -    | 99.8 | 92.8   | 90.2         | -        |
| F$^2$PN    | TPAMI 2022  | Prediction             | 84.3 | 96.2 | 85.7   | 73.0         | -        |
| DLAN-AC    | ECCV 2022   | Reconstruction         | -    | 97.6 | 89.9   | 74.7         | -        |
| BDPN       | AAAI 2022   | Prediction             | -    | 98.3 | 90.3   | 78.1         | -        |
| CAFÉ       | ACMMM 2022  | Prediction             | -    | 98.4 | 92.6   | 77.0         | -        |
| STJP       | ECCV 2022   | Jigsaw puzzle          | -    | 99.0 | 92.2   | 84.3         | 56.4     |
| MPT        | ICCV 2023   | Multiple tasks         | -    | 97.6 | 90.9   | 78.8         | -        |
| HSC        | CVPR 2023   | Hybrid                 | -    | 98.1 | 93.7   | 83.4         | -        |
| LERF       | AAAI 2023   | Predicition            | -    | 99.4 | 91.5   | 78.6         | -        |
| DMAD       | CVPR 2023   | Reconstruction         | -    | 99.7 | 92.8   | 78.8         | -        |
| EVAL       | CVPR 2023   | Interpretable learning | -    | -    | 86.0   | 76.6         | -        |
| FBSC-AE    | CVPR 2023   | Prediction             | -    | -    | 86.8   | 79.2         | -        |
| FPDM       | ICCV 2023   | Prediction             | -    | -    | 90.1   | 78.6         | 62.7     |
| PFMF       | CVPR 2023   | Multiple tasks         | -    | -    | 93.6   | 85.0         | -        |
| STG-NF     | ICCV 2023   | Gaussian classifier    | -    | -    | -      | 85.9         | 71.8     |
| AED-MAE    | CVPR 2024   | Patch inpainting       | -    | 95.4 | 91.3   | 79.1         | 58.5     |
| SSMCTB     | TPAMI 2024  | Patch inpainting       | -    | -    | 91.6   | 83.7         | -        |

- **Quantitative Performance Comparison of Weakly Supervised Methods on Public Datasets.**
  
  | Method   | Publication | Feature        | UCF-Crime | XD-Violence | ShanghaiTech | TAD   |
  | -------- | -----------:| -------------- |:---------:|:-----------:|:------------:|:-----:|
  | DeepMIL  | CVPR 2018   | C3D(RGB)       | 75.40     | -           | -            | -     |
  | GCN      | CVPR 2019   | TSN(RGB)       | 82.12     | -           | 84.44        | -     |
  | HLNet    | ECCV 2020   | I3D(RGB)       | 82.44     | 75.41       | -            | -     |
  | CLAWS    | ECCV 2020   | C3D(RGB)       | 83.03     | -           | 89.67        | -     |
  | MIST     | CVPR 2021   | I3D(RGB)       | 82.30     | -           | 94.83        | -     |
  | RTFM     | ICCV 2021   | I3D(RGB)       | 84.30     | 77.81       | 97.21        | -     |
  | CTR      | TIP 2021    | I3D(RGB)       | 84.89     | 75.90       | 97.48        | -     |
  | MSL      | AAAI 2022   | VideoSwin(RGB) | 85.62     | 78.59       | 97.32        | -     |
  | S3R      | ECCV 2022   | I3D(RGB)       | 85.99     | 80.26       | 97.48        | -     |
  | SSRL     | ECCV 2022   | I3D(RGB)       | 87.43     | -           | 97.98        | -     |
  | CMRL     | CVPR 2023   | I3D(RGB)       | 86.10     | 81.30       | 97.60        | -     |
  | CUPL     | CVPR 2023   | I3D(RGB)       | 86.22     | 81.43       | -            | 91.66 |
  | MGFN     | AAAI 2023   | VideoSwin(RGB) | 86.67     | 80.11       | -            | -     |
  | UMIL     | CVPR 2023   | CLIP           | 86.75     | -           | -            | 92.93 |
  | DMU      | AAAI 2023   | I3D(RGB)       | 86.97     | 81.66       | -            | -     |
  | PE-MIL   | CVPR 2024   | I3D(RGB)       | 86.83     | 88.05       | 98.35        | -     |
  | TPWNG    | CVPR 2024   | CLIP           | 87.79     | 83.68       | -            | -     |
  | VadCLIP  | AAAI 2024   | CLIP           | 88.02     | 84.51       | -            | -     |
  | STPrompt | ACMMM 2024  | CLIP           | 88.08     | -           | 97.81        | -     |

- **Quantitative Performance Comparison of Fully Supervised Methods on Public Datasets.**
  
  | Method       | Publication | Model Input               | Hockey Fights | Violent-Flows | RWF-2000 | Crowed Violence |
  | ------------ | ----------- | ------------------------- | ------------- | ------------- | -------- | --------------- |
  | TS-LSTM      | PR 2016     | RGB+Flow                  | 93.9          | -             | -        | -               |
  | FightNet     | JPCS 2017   | RGB+Flow                  | 97.0          | -             | -        | -               |
  | ConvLSTM     | AVSS 2017   | Frame Difference          | 97.1          | 94.6          | -        | -               |
  | BiConvLSTM   | ECCVW 2018  | Frame Difference          | 98.1          | 96.3          | -        | -               |
  | SPIL         | ECCV 2020   | Skeleton                  | 96.8          | -             | 89.3     | 94.5            |
  | FlowGatedNet | ICPR 2020   | RGB+Flow                  | 98.0          | -             | 87.3     | 88.9            |
  | X3D          | AVSS 2022   | RGB                       | -             | 98.0          | 94.0     | -               |
  | HSCD         | CVIU 2023   | Skeleton+Frame Difference | 94.5          | -             | 90.3     | 94.3            |

- **Quantitative Performance Comparison of Unsupervised Methods on Public Datasets.**
  
  | Method    | Publication | Methodology      | Avenue | Subway Exit | Ped1 | Ped2 | ShaihaiTech | UMN  |
  | --------- | ----------- | ---------------- | ------ | ----------- | ---- | ---- | ----------- | ---- |
  | ADF       | ECCV 2016   | Change detection | 78.3   | 82.4        | -    | -    | -           | 91.0 |
  | Unmasking | ICCV 2017   | Change detection | 80.6   | 86.3        | 68.4 | 82.2 | -           | 95.1 |
  | MC2ST     | BMVC 2018   | Change detection | 84.4   | 93.1        | 71.8 | 87.5 | -           | -    |
  | DAW       | ACMMM 2018  | Pseudo label     | 85.3   | 84.5        | 77.8 | 96.4 | -           | -    |
  | STDOR     | CVPR 2020   | Pseudo label     | -      | 92.7        | 71.7 | 83.2 | -           | 97.4 |
  | TMAE      | ICME 2022   | Change detection | 89.8   | -           | 75.7 | 94.1 | 71.4        | -    |
  | CIL       | AAAI 2022   | Others           | 90.3   | 97.6        | 84.9 | 99.4 | -           | 100  |
  | LBR-SPR   | CVPR 2022   | Others           | 92.8   | -           | 81.1 | 97.2 | 72.6        | -    |

## Citation

If you find our work useful, please cite our paper:

```
@article{wu2024deep,
  title={Deep Learning for Video Anomaly Detection: A Review},
  author={Wu, Peng and Pan, Chengyu and Yan, Yuting and Pang, Guansong and Wang, Peng and Zhang, Yanning},
  journal={arXiv preprint arXiv:xxxxx},
  year={2024}
}
```
