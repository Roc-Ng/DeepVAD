- [Dataset Information](#dataset-information)
  - [VAD Datasets Table](#vad-datasets-table)
  - [Video Anomaly Caption/Retrieval Table](#video-anomaly-captionretrieval-table)

- [1. Dataset](#1-dataset)  
  - [1.1 Weakly supervised VAD](#11-weakly-supervised-vad)  
    - [1.1.1 UCF-Crime (CVPR 2018)](#111-ucf-crimereal-world-anomaly-detection-in-surveillance-videoscvpr-2018)  
    - [1.1.2 ShanghaiTech Weakly (CVPR 2019)](#112-shanghaitech-weaklygraph-convolutional-label-noise-cleaner-train-a-plug-and-play-action-classifier-for-anomaly-detectioncvpr-2019)  
    - [1.1.3 XD-Violence (ECCV 2020)](#113-xd-violencenot-only-look-but-also-listen-learning-multimodal-violence-detection-under-weak-supervisioneccv-2020)  
    - [1.1.4 TAD Weakly (TIP 2021)](#114-tad-weaklylocalizing-anomalies-from-weakly-labeled-videostip-2021)  
  - [1.2 Semi-supervised VAD](#12-semi-supervised-vad)  
    - [1.2.1 & 1.2.2 Subway Entrance & Exit (TPAMI 2008)](#121122-subway-entranceexitrobust-real-time-unusual-event-detection-using-multiple-fixed-location-monitorstpami-2008)  
    - [1.2.3 UMN (CVPR 2009)](#123-umnabnormal-crowd-behavior-detection-using-social-force-modelcvpr-2009)  
    - [1.2.4 & 1.2.5 UCSD Ped1 & Ped2 (CVPR 2010)](#124--125-ucsd-ped1--ped2anomaly-detection-in-crowded-scenescvpr-2010)  
    - [1.2.6 CUHK Avenue (ICCV 2013)](#126-cuhk-avenueabnormal-event-detection-at-150-fps-in-matlabiccv-2013)  
    - [1.2.7 ShanghaiTech (ICCV 2017)](#127-shanghaitecha-revisit-of-sparse-coding-based-anomaly-detection-in-stacked-rnn-frameworkiccv-2017)  
    - [1.2.8 Street Scene (WACV 2020)](#128-street-scenestreet-scene-a-new-dataset-and-evaluation-protocol-for-video-anomaly-detectionwacv-2020)  
    - [1.2.9 IITB-Corridor (WACV 2020)](#129-iitb-corridormulti-timescale-trajectory-prediction-for-abnormal-human-activity-detectionwacv-2020)  
    - [1.2.10 NWPU Campus (CVPR 2023)](#1210-nwpu-campusa-new-comprehensive-benchmark-for-semi-supervised-video-anomaly-detection-and-anticipationcvpr-2023)  
  - [1.3 Fully Supervised VAD](#13-fully-supervised-vad)  
    - [1.3.1 & 1.3.2 Hockey Fight & Movies Fight (CAIP 2011)](#131132-hockey-fight--movies-fightviolence-detection-in-video-using-computer-vision-techniquescaip-2011)  
    - [1.3.3 Violent-Flows (CVPRW 2012)](#133-violent-flowsviolent-flows-real-time-detection-of-violent-crowd-behaviorcvpr-workshops-2012)  
    - [1.3.4 VSD (MTA 2015)](#134-vsdvsd-a-public-dataset-for-the-detection-of-violent-scenes-in-movies-design-annotation-analysis-and-evaluationmta-2015)  
    - [1.3.5 CCTV-Fights (ICASSP 2019)](#135-cctv-fightsdetection-of-real-world-fights-in-surveillance-videosicassp-2019)  
    - [1.3.6 RWF-2000 (ICPR 2020)](#136-rwf-2000rwf-2000-an-open-large-scale-video-database-for-violence-detectionicpr-2020)  
    - [1.3.7 VFD-2000 (ICTAI 2022)](#137-vfd-2000weakly-supervised-two-stage-training-scheme-for-deep-video-fight-detection-modelictai-2022)  
  - [1.4 Open-set Supervised VAD](#14-open-set-supervised-vad)  
    - [1.4.1 UBnormal (CVPR 2022)](#141-ubnormalubnormal-new-benchmark-for-supervised-open-set-video-anomaly-detectioncvpr-2022)  
  - [1.5 Video Anomaly Caption/Retrieval](#15-video-anomaly-captionretrieval)  
    - [1.5.1 UCA (CVPR 2024)](#151-ucatowards-surveillance-video-and-language-understanding-new-dataset-baselines-and-challengescvpr-2024)  
    - [1.5.2 VAD-Instruct50k (arXiv 2024)](#152-vad-instruct50kholmes-vad-towards-unbiased-and-explainable-video-anomaly-detection-via-multi-modal-llm)  
    - [1.5.3 & 1.5.4 UCFCrimeAR & XDViolenceAR (TIP 2024)](#ucfcrimear-xdviolencear)  
    - [1.5.5 UCCD (TMM 2024)](#155-uccdhuman-centric-behavior-description-in-videos-new-benchmark-and-modeltmm-2024)  
- [2. Performance Evaluation](#2-performance-evaluation)
  - [2.1 Frame-level metrics](#21-frame-level-metrics)
    - [2.1.1 AUC (Area Under the Curve)](#211-auc-area-under-the-curvepaperframe-level-aucpixel-level-auc)
    - [2.1.2 AP (Average Precision)](#212-ap-average-precisionpapermap)
    - [2.1.3 EER (Equal Error Rate) and EDR (Equal Detected Rate)](#213-eer-equal-error-rate-and-edr-equal-detected-ratepaper)
    - [2.1.4 Accuracy](#214-accuracypaper)
  - [2.2 Pixel-level metrics](#22-pixel-level-metrics)
    - [2.2.1 TIoU (Temporal Intersection over Union)](#221-tiou-temporal-intersection-over-unionpaper)
    - [2.2.2 RBDR (Region-based Detection Rate) and TBDR (Track-based Detection Rate)](#222-rbdr-region-based-detection-rate-and-tbdr-track-based-detection-ratepaper)
  - [2.3 Sentence-level and reasoning-based metrics](#23-sentence-level-and-reasoning-based-metrics)
    - [2.3.1 BLEU (Bilingual Evaluation Understudy)](#231-bluebilingual-evaluation-understudypaper)
    - [2.3.2 CIDEr (Consensus-based Image Description Evaluation)](#232-ciderconsensus-based-image-description-evaluationpaper)
    - [2.3.3 METEOR (Metric for Evaluation of Translation with Explicit ORdering)](#233-meteormetric-for-evaluation-of-translation-with-explicit-orderingpaper)
    - [2.3.4 ROUGE (Recall-Oriented Understudy for Gisting Evaluation)](#234-rougerecall-oriented-understudy-for-gisting-evaluationpaper)
    - [2.3.5 MMEval](#235-mmevalpaper)


    
#  Dataset Information
## VAD Datasets Table

|       **Number**        | **Year** |                         **Dataset**                          | **Videos** |  Videos  | Videos | Videos |  Videos  | **Frames** |  Frames   | Frames  |  Frames   |  Frames  | **Hours** |
| :---------------------: | :------: | :----------------------------------------------------------: | :--------: | :------: | :----: | :----: | :------: | :--------: | :-------: | :-----: | :-------: | :------: | :-------: |
|                         |          |                                                              |   Total    | Training |  Test  | Normal | Abnormal |   Total    | Training  |  Test   |  Normal   | Abnormal |           |
|  1.1 Weakly Supervised VAD  |          |                                                              |            |          |        |        |          |            |           |         |           |          |           |
|            1.1.1            |   2018   | [UCF\-Crime](https://www.crcv.ucf.edu/projects/real-world/#:~:text=We%20construct%20a%20new%20large,Stealing%2C%20Shoplifting%2C%20and%20Vandalism.) |   1,900    |  1,610   |  290   |  950   |   950    |     -      |     -     |    -    |     -     |    -     |    128    |
|            1.1.2            |   2019   | [ShanghaiTech Weakly](https://github.com/jx-zhong-for-academic-purpose/GCN-Anomaly-Detection/tree/master/ShanghaiTech_new_split) |    437     |   238    |  199   |  330   |   107    |     -      |     -     |    -    |     -     |    -     |     -     |
|            1.1.3            |   2020   |    [XD\-Violence](https://roc-ng.github.io/XD-Violence/)     |   4,754    |  3,954   |  800   | 2,349  |  2,405   |     -      |     -     |    -    |     -     |    -     |    217    |
|            1.1.4            |   2021   |       [TAD Weakly](https://github.com/ktr-hubrt/WSAL)        |    500     |   400    |  100   |  250   |   250    |  540,212   |     -     |    -    |     -     |    -     |     -     |
|  1.2 Semi\-supervised VAD   |          |                                                              |            |          |        |        |          |            |           |         |           |          |           |
|            1.2.1            |   2008   | [Subway Entrance](https://vision.eecs.yorku.ca/research/anomalous-behaviour-data/sets/) |     1      |    -     |   -    |   -    |    -     |  144,249   |     -     |    -    |     -     |    -     |    1.6    |
|            1.2.2            |   2008   | [Subway Exit](https://vision.eecs.yorku.ca/research/anomalous-behaviour-data/sets/) |     1      |    -     |   -    |   -    |    -     |   64,901   |     -     |    -    |     -     |    -     |    0.7    |
|            1.2.3            |   2009   | [UMN](https://www.crcv.ucf.edu/research/projects/abnormal-crowd-behavior-detection-using-social-force-model/) |     5      |    -     |   -    |   -    |    -     |  7,741   |     -     |    -    |  6,165  | 1,576  |     -     |
|            1.2.4            |   2010   | [UCSD Ped1](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm) |     70     |    34    |   36   |   34   |    36    |  14,000  |   6,800   |  7,200  |   9,995   |  4,005   |     -     |
|            1.2.5            |   2010   | [UCSD Ped2](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm) |     28     |    16    |   12   |   16   |    12    |  4,560   |   2,550   |  2,010  |   2,924   |  1,636   |     -     |
|            1.2.6            |   2013   | [CUHK Avenue](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html) |     37     |    16    |   21   |   16   |    21    |   30,652   |  15,328   | 15,324  |  26,832   |  3,820   |    0.5    |
|            1.2.7            |   2017   | [ShanghaiTech](https://svip-lab.github.io/dataset/campus_dataset.html) |    437     |   330    |  107   |  330   |   107    |  317,398   |  274,515  | 42,883  |  300,308  |  17,090  |     -     |
|            1.2.8            |   2020   | [Street Scene](https://www.merl.com/demos/video-anomaly-detection) |     81     |    46    |   35   |   46   |    35    |  203,257   |  56,847   | 146,410 |  159,341  |  43,916  |     -     |
|            1.2.9            |   2020   | [IITB-Corridor](https://rodrigues-royston.github.io/Multi-timescale_Trajectory_Prediction/) |     358     |    208    |   150   |    -    |     -     |  483,566   |  301,999  | 181,567 |  375,288  | 108,278  |     -      |
|           1.2.10            |   2023   |         [NWPU Campus](https://campusvad.github.io/)          |    547     |   305    |  242   |  305   |   242    | 1,466,073  | 1,082,014 | 384,059 | 1,400,807 |  65,266  |   16.3    |
|  1.3 Fully Supervised VAD   |          |                                                              |            |          |        |        |          |            |           |         |           |          |           |
|            1.3.1            |   2011   | [Hockey Fight](https://www.kaggle.com/datasets/yassershrief/hockey-fight-vidoes) |   1,000    |    -     |   -    |  500   |   500    |     -      |     -     |    -    |     -     |    -     |     -     |
|            1.3.2            |   2011   | [Movies Fight](https://academictorrents.com/details/70e0794e2292fc051a13f05ea6f5b6c16f3d3635) |    200     |    -     |   -    |  100   |   100    |     -      |     -     |    -    |     -     |    -     |     -     |
|            1.3.3            |   2012   | [Violent\-Flows](https://www.openu.ac.il/home/hassner/data/violentflows/) |    246     |    -     |   -    |   -    |    -     |     -      |     -     |    -    |     -     |    -     |    0.2    |
|            1.3.4            |   2015   | [VSD](https://www.interdigital.com/data_sets/violent-scenes-dataset) |     18     |    15    |   3    |   -    |    -     |     -      |     -     |    -    |     -     |    -     |    35     |
|            1.3.5            |   2019   | [CCTV\-Fights](https://rose1.ntu.edu.sg/dataset/cctvFights/) |   1,000    |   500    |  250   |   -    |    -     |     -      |     -     |    -    |     -     |    -     |    18     |
|            1.3.6            |   2020   | [RWF\-2000](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection) |   2,000    |  1,600   |  400   | 1,000  |  1,000   |     -      |     -     |    -    |     -     |    -     |     3     |
|           1.3.7            |   2022   | [VFD\-2000](https://github.com/Hepta-Col/VideoFightDetection) |   2,490    |    -     |   -    | 1,194  |  1,296   |     -      |     -     |    -    |     -     |    -     |     -     |
| 1.4 Open-set Supervised VAD |          |                                                              |            |          |        |        |          |            |           |         |           |          |           |
|            1.4.1            |   2022   |    [UBnormal](https://github.com/lilygeorgescu/UBnormal)     |    543     |   268    |  211   |   -    |    -     |  236,902   |  116,087  | 92,640  |  147,887  |  89,015  |    2.2    |

## Video Anomaly Caption/Retrieval Table

| **Number** | **Year** |                         **Dataset**                          | **Videos** | **Queries** | **Avg word** | Hours |
| :--------: | :------: | :----------------------------------------------------------: | :--------: | :---------: | :----------: | :---: |
|     1.5.1      |   2024   | [UCA](https://xuange923.github.io/Surveillance-Video-Understanding) |   1,854    |   23,542    |     20.2     |  111  |
|     1.5.2      |   2024   |       [VAD-Instruct50k](https://holmesvad.github.io/)        |   5,547    |   51,567    |     44.8     |   -   |
|     1.5.3      |   2024   | [UCFCrimeAR](https://github.com/Roc-Ng/VAR?tab=readme-ov-file) |   1,900    |    1,900    |     16.3     |  128  |
|     1.5.4      |   2024   | [XDViolenceAR](https://github.com/Roc-Ng/VAR?tab=readme-ov-file) |   4,754    |    4,754    |      -       |  217  |
|     1.5.5      |   2024   |          [UCCD](https://github.com/lingruzhou/UCCD)          |   1,012    |    7,820    |     34.0     |  112  |

# 1. Dataset

## 1.1 Weakly supervised VAD

### 1.1.1 UCF-Crime:Real-world Anomaly Detection in Surveillance Videos(CVPR 2018)

[[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.pdf)][[Homepage](https://www.crcv.ucf.edu/projects/real-world/#:~:text=We%20construct%20a%20new%20large,Stealing%2C%20Shoplifting%2C%20and%20Vandalism.)]

<p align = "justify"> 
The dataset capture 13 realistic anomalies such as fighting, road accidents, burglary, robbery, and other illegal activities, along with normal activities.The videos were collected from YouTube and LiveLeak using text search queries in various languages to ensure a diverse set of anomalies. The collection process involved stringent criteria to exclude videos that were manually edited, pranks, not captured by CCTV cameras, news footage, handheld camera recordings, or compilations.
</p>

<p align = "center">
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%871.png" width="900" />
</p>

### 1.1.2 ShanghaiTech Weakly:Graph Convolutional Label Noise Cleaner: Train a Plug-And-Play Action Classifier for Anomaly Detection(CVPR 2019)

[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhong_Graph_Convolutional_Label_Noise_Cleaner_Train_a_Plug-And-Play_Action_Classifier_CVPR_2019_paper.pdf)][[Homepage](https://github.com/jx-zhong-for-academic-purpose/GCN-Anomaly-Detection/tree/master/ShanghaiTech_new_split)]

<p align = "justify"> 
The dataset includes 130 abnormal events across 13 different scenes. It is considered a medium-scale dataset compared to other datasets used in the study. In the standard protocol for the ShanghaiTech dataset, all training videos are normal. However, for the binary-classification task and the weakly supervised setting, the authors reorganize the dataset by randomly selecting anomaly testing videos into the training data and vice versa.
</p>
<p align = "center">
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%872.png" width="900" />
</p>

### 1.1.3 XD-Violence:Not only Look, but also Listen: Learning Multimodal Violence Detection under Weak Supervision(ECCV 2020)

[[Paper](https://link.springer.com/chapter/10.1007/978-3-030-58577-8_20)][[Homepage](https://roc-ng.github.io/XD-Violence/)]

<p align = "justify"> 
The dataset is a large-scale, multi-scene, multi-modal dataset designed for the purpose of violence detection in videos. The dataset consists of a total duration of 217 hours, containing untrimmed videos with audio signals and weak labels.The dataset includes six violent classes. They are Abuse, Car Accident, Explosion, Fighting, Riot, and Shooting. The videos were collected from both movies and YouTube (in-the-wild scenes). The authors make an effort to collect non-violent videos whose background is consistent with that of violent videos to prevent discrimination based on scenario backgrounds. The annotations are made more precise by assigning the same videos to multiple annotators and averaging their annotations.
</p>
<p align = "center">
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%873.1.png" width="225" />
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%873.2.png" width="225" />
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%873.3.png" width="225" />
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%873.4.png" width="225" />
</p>

### 1.1.4 TAD Weakly:Localizing Anomalies From Weakly-Labeled Videos(TIP 2021)

[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9408419)][[Homepage](https://github.com/ktr-hubrt/WSAL)]

<p align = "justify"> 
The dataset is a large-scale collection of surveillance videos designed for the purpose of video anomaly detection, with a particular focus on traffic scenarios.The TAD dataset consists of long, untrimmed videos that capture various real-world anomalies occurring in traffic scenes. It is comprehensive and includes videos from different scenarios, weather conditions, and times of day.The traffic videos in the TAD dataset were collected from various countries, downloaded from platforms like YouTube or Google, and were primarily recorded by CCTV cameras mounted on roads. The collection process excluded videos that were manually edited, pranks, or compilations, as well as those with ambiguous anomalies.The dataset covers seven real-world anomalies on roads, such as Vehicle Accidents, Illegal Turns, Illegal Occupations, Retrograde Motion, Pedestrian on Road, Road Spills, and a category for other anomalies known as "The Else."
</p>
<p align = "center">
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%874.png" width="900" />
</p>

## 1.2 Semi-supervised VAD

### 1.2.1&1.2.2 Subway Entrance&Exit:Robust Real-Time Unusual Event Detection using Multiple Fixed-Location Monitors(TPAMI 2008)

[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4407716)][[Homepage](https://vision.eecs.yorku.ca/research/anomalous-behaviour-data/sets/)]

<p align = "justify"> 
The Subway Entrance and Exit datasets are video datasets that have been used for the purpose of unusual event detection in surveillance videos.The datasets consist of surveillance videos captured at subway entrances and exits. The videos contain a variety of events, including normal pedestrian traffic and potentially unusual activities.
</p>
<p align = "center">
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%875.png" width="900" />
</p>
<p align = "center">
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%876.png" width="900" />
</p>

### 1.2.3 UMN:Abnormal crowd behavior detection using social force model(CVPR 2009)

[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5206641)][[Homepage](https://www.crcv.ucf.edu/research/projects/abnormal-crowd-behavior-detection-using-social-force-model/)]

<p align = "justify"> 
The dataset is a collection of video sequences designed for the purpose of unusual crowd activity detection. The dataset comprises 11 short videos of 3 different scenarios depicting an escape event in various indoor and outdoor settings. The videos include both normal behavior at the beginning and abnormal behavior towards the end.Each video in the dataset consists of a sequence that starts with normal crowd behavior and transitions into abnormal behavior. 
</p>
<p align = "center">
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%877.1.png" width="450" />
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%877.2.png" width="450" />
</p>

### 1.2.4&1.2.5 UCSD Ped1 & Ped2:Anomaly detection in crowded scenes(CVPR 2010)

[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5539872)][[Homepage](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm)]

<p align = "justify"> 
The UCSD Ped1 & Ped2 datasets are video datasets created for the purpose of anomaly detection in crowded scenes. The datasets consist of video sequences captured from a stationary camera overlooking pedestrian walkways. The scenes include varying densities of crowds, ranging from sparse to very crowded.Anomalies in the dataset can be the circulation of non-pedestrian entities in the walkways or anomalous pedestrian motion patterns, such as bikers, skaters, small carts, and people walking across a walkway or in the grass surrounding it.All anomalies in the dataset are naturally occurring and were not staged for the purposes of assembling the dataset.The data was split into two subsets, each corresponding to a different scene. The first scene contains groups of people walking towards and away from the camera, with some perspective distortion. The second scene contains pedestrian movement parallel to the camera plane. Additionally, a subset of 10 clips is provided with manually generated pixel-level binary masks that identify the regions containing anomalies.
</p>
<p align = "center">
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%878.png" width="900" />
</p>

### 1.2.6 CUHK Avenue:Abnormal Event Detection at 150 FPS in MATLAB(ICCV 2013)

[[Paper](https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Lu_Abnormal_Event_Detection_2013_ICCV_paper.pdf)][[Homepage](https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)]

<p align = "justify"> 
The dataset is a collection of video sequences designed for the purpose of abnormal event detection in surveillance videos. The dataset contains 15 sequences, each about 2 minutes long, totaling 35,240 frames. It includes various unusual events such as running, throwing objects, and loitering. Four videos are used as training data, comprising 8,478 frames in total.
</p>
<p align = "center">
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%879.png" width="900" />
</p>

### 1.2.7 ShanghaiTech:A Revisit of Sparse Coding Based Anomaly Detection in Stacked RNN Framework(ICCV 2017)

[[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Luo_A_Revisit_of_ICCV_2017_paper.pdf)][[Homepage](https://svip-lab.github.io/dataset/campus_dataset.html)]

<p align = "justify"> 
The dataset is very large in both the volume of data and the diversity of scenes.Unlike many existing datasets that contain videos captured by a single fixed camera, the ShanghaiTech dataset includes videos from multiple surveillance cameras installed at different spots with varying view angles. This captures a wider range of real-world scenarios.The dataset captures real events that happened in the living area of the university campus, including sudden motions such as chasing and brawling.
</p>
<p align = "center">
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%8710.png" width="900" />
</p>

### 1.2.8 Street Scene:Street Scene: A new dataset and evaluation protocol for video anomaly detection(WACV 2020)

[[Paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Ramachandra_Street_Scene_A_new_dataset_and_evaluation_protocol_for_video_WACV_2020_paper.pdf)][[Homepage](https://www.merl.com/research/highlights/video-anomaly-detection)]

<p align = "justify"> 
The dataset is a large and varied video dataset. The dataset was captured from a static camera overlooking a two-lane urban street, including bike lanes and pedestrian sidewalks. The testing sequences contain a total of 205 anomalous events consisting of 17 different types of anomalies. These include jaywalking, loitering, illegal U-turns, and bikers or cars outside their lanes, among others.The authors aimed to include only "natural" anomalies in the dataset, meaning the anomalies were not staged by actors but occurred organically in the surveillance footage.
</p>
<p align = "center">
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%8711.1.png" width="225" />
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%8711.2.png" width="225" />
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%8711.3.png" width="225" />
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%8711.4.png" width="225" />
</p>

### 1.2.9 IITB-Corridor:Multi-timescale Trajectory Prediction for Abnormal Human Activity Detection(WACV 2020)

[[Paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Rodrigues_Multi-timescale_Trajectory_Prediction_for_Abnormal_Human_Activity_Detection_WACV_2020_paper.pdf)][[Homepage](https://rodrigues-royston.github.io/Multi-timescale_Trajectory_Prediction/)]

<p align = "justify"> 
The IITB-Corridor Dataset is a single-camera dataset introduced for the purpose of abnormal human activity detection. It was created at the Indian Institute of Technology Bombay. The videos are captured using a single camera, which makes the dataset more challenging and realistic compared to multi-camera setups. The dataset includes a range of activities, from normal activities like walking and standing to various abnormal activities such as loitering, sudden running, fighting, chasing, and more. The dataset provides annotations at the frame level, distinguishing between normal and abnormal activities. It contains not only single-person anomalies but also multiple-person and group-level anomalies, making it a comprehensive resource for studying different types of abnormal behaviors.
</p>

<p align = "center">
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%871.2.9.png" width="900" />
</p>

### 1.2.10 NWPU Campus:A New Comprehensive Benchmark for Semi-supervised Video Anomaly Detection and Anticipation(CVPR 2023)

[[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Cao_A_New_Comprehensive_Benchmark_for_Semi-Supervised_Video_Anomaly_Detection_and_CVPR_2023_paper.pdf)][[Homepage](https://campusvad.github.io/)]

<p align = "justify"> 
The dataset is a new comprehensive benchmark introduced for semi-supervised video anomaly detection (VAD) and video anomaly anticipation (VAA). The dataset is designed to address the lack of scene-dependent anomalies and the absence of a suitable dataset for anomaly anticipation in existing research.The NWPU Campus dataset is the largest semi-supervised VAD dataset to date, containing 43 scenes, 28 classes of abnormal events, and 16 hours of video footage. It surpasses the previous largest dataset, IITB Corridor, in terms of size and duration. It is the only dataset that considers scene-dependent anomalies, which are events that are normal in one context but abnormal in another (e.g., playing football on the playground is normal, but playing on the road is abnormal). Unlike some other datasets that use animated or simulated scenarios, the NWPU Campus dataset consists of real recorded videos, making it more reflective of real-world conditions. The dataset includes a wide range of abnormal events, such as single-person anomalies, interaction anomalies, group anomalies, location anomalies, appearance anomalies, and trajectory anomalies.It is the first dataset proposed for video anomaly anticipation, which is the task of predicting the occurrence of abnormal events in advance based on the trend of the event. This is significant for early warning systems to prevent dangerous accidents.The dataset was collected by setting up cameras at 43 outdoor locations on a university campus, capturing activities of pedestrians and vehicles.
</p>
<p align = "center">
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%8713.png" width="900" />
</p>

## 1.3 Fully Supervised VAD

### 1.3.1&1.3.2 Hockey Fight & Movies Fight:Violence Detection in Video Using Computer Vision Techniques(CAIP 2011)

[[Paper](https://www.cs.cmu.edu/~rahuls/pub/caip2011-rahuls.pdf)][[Homepage](https://academictorrents.com/details/70e0794e2292fc051a13f05ea6f5b6c16f3d3635)]

<p align = "justify"> 
The dataset is divided into two parts: the "Hockey Fight" dataset and the "Movies Fight" dataset.
Hockey Fight Dataset was taken from National Hockey League (NHL) hockey games.The clips are manually labeled as either "fight" or "non-fight," providing a clear distinction for training and testing violence detection models.The uniformity in format and content, along with the dynamic settings where both normal and violent activities occur, make it suitable for measuring the performance of various violence recognition approaches robustly.
Movies Fight Dataset is composed of 200 video clips from action movies, out of which 100 contain a fight.Unlike the hockey dataset, the action movie clips depict a wider variety of scenes and are captured at different resolutions, making it more challenging for the detection models due to the variability in cinematography and appearance.The fight scenes in movies are more varied, and the videos may contain different camera angles and motions.
</p>
<p align = "center">
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%8715.1.png" width="450" />
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%8715.2.png" width="450" />
</p>

<p align = "center">
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%8716.1.png" width="450" />
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%8716.2.png" width="450" />
</p>

### 1.3.3 Violent-Flows:Violent flows: Real-time detection of violent crowd behavior(CVPR Workshops 2012)

[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6239348)][[Homepage](https://www.openu.ac.il/home/hassner/data/violentflows/)]

<p align = "justify"> 
The dataset is a unique collection of real-world surveillance videos designe for the detection of violent behavior in crowded scenes.The dataset is downloaded from the web, representing a wide range of real-world, unconstrained conditions and scenarios. The videos depict both violent and non-violent crowd behaviors and are intended to reflect the variability and complexity of real-world surveillance footage.The videos are sourced from YouTube, ensuring a diverse set of scenarios and conditions.
</p>
<p align = "center">
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%8718.1.png" width="450" />
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%8718.2.png" width="225" />
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%8718.3.png" width="225" />
</p>

### 1.3.4 VSD:VSD, a public dataset for the detection of violent scenes in movies: design, annotation, analysis and evaluation(MTA 2015)

[[Paper](https://link.springer.com/article/10.1007/s11042-014-1984-4)][[Homepage](https://www.interdigital.com/data_sets/violent-scenes-dataset)]

<p align = "justify"> 
The dataset is designed for the development of content-based detection techniques targeting physical violence in Hollywood movies.The VSD dataset consists of 18 movies, chosen to provide a diverse range of genres and types of violence. The movies include extremely violent ones like "Kill Bill" or "Fight Club" and others with virtually no violent content like "The Wizard of Oz". The dataset includes rich annotations beyond the annotation of violent segments. It encompasses the presence of blood, fights, fire, guns, cold weapons, car chases, gory scenes, gunshots, explosions, and screams.
</p>
<p align = "center">
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%8720.1.png" width="675" />
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%8720.2.png" width="225" />
</p>

### 1.3.5 CCTV-Fights:Detection of Real-world Fights in Surveillance Videos(ICASSP 2019)

[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8683676)][[Homepage](https://rose1.ntu.edu.sg/dataset/cctvFights/)]

<p align = "justify"> 
The dataset is a novel and challenging collection of videos designed to facilitate the development and evaluation of automated solutions for detecting real-world fights in surveillance footage.The videos depict a diverse range of actions and attributes, such as punching, kicking, pushing, and wrestling, involving two or more persons. The dataset includes fights recorded from CCTV cameras as well as from mobile cameras, car cameras (dash-cams), and drones or helicopters.The videos were collected from YouTube and other sources using keywords related to real fights and surveillance.The dataset includes both CCTV and non-CCTV videos, with the CCTV videos being longer in duration (average length of 2 minutes) and the non-CCTV videos being shorter (average length of 45 seconds).The videos are annotated at the frame level.The dataset is split into 50% for training, 25% for validation, and 25% for testing.
</p>
<p align = "center">
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%8721.1.png" width="450" />
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%8721.2.png" width="450" />
</p>

### 1.3.6 RWF-2000:RWF-2000: An Open Large Scale Video Database for Violence Detection(ICPR 2020)

[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9412502)][[Homepage](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection)]

<p align = "justify"> 
The dataset is a large-scale video database.The RWF-2000 dataset depict real-world violent scenes, such as fights, robberies, explosions, and assaults. The videos are captured by various devices, including mobile cameras, car-mounted cameras, and other surveillance equipment. The dataset provides a diverse range of violent activities and attributes, making it suitable for training and testing violence detection models. The videos are annotated as either "Violent" or "Non-Violent," providing a clear distinction for training and evaluation purposes.The dataset includes frame-level annotations for violent activities.
</p>
<p align = "center">
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%8722.1.png" width="675" />
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%8722.2.png" width="225" />
</p>

### 1.3.7 VFD-2000:Weakly Supervised Two-Stage Training Scheme for Deep Video Fight Detection Model(ICTAI 2022)

[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10098024)][[Homepage](https://github.com/Hepta-Col/VideoFightDetection)]

<p align = "justify"> 
The dataset is a large-scale, multi-scenario video dataset. It was created to address the limitations of other datasets, which often suffer from small scale, limited scenarios, and fixed video lengths. The videos cover a wide range of scenarios and include both fight and non-fight behaviors. The dataset is annotated with video-level and frame-level labels. The dataset includes videos recorded in various real-life situations, such as street fights, violence in restaurants, and other confrontations. It contains videos recorded by different devices, including smartphones and surveillance cameras, resulting in a diverse set of recording conditions. The videos are categorized into four groups based on their length and view orientation: long vertical view, short vertical view, long horizontal view, and short horizontal view. The dataset includes ambiguous behavior clips that show actions between fight and non-fight, providing a more realistic and challenging set of data for training models.The dataset features manual video-level labeling for each clip, with additional frame-level labels for long videos in the test set. 
</p>
<p align = "center">
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%8724.1.png" width="450" />
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%8724.2.png" width="450" />
</p>

## 1.4 Open-set Supervised VAD

### 1.4.1 UBnormal:UBnormal: New Benchmark for Supervised Open-Set Video Anomaly Detection(CVPR 2022)

[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Acsintoae_UBnormal_New_Benchmark_for_Supervised_Open-Set_Video_Anomaly_Detection_CVPR_2022_paper.pdf)][[Homepage](https://github.com/lilygeorgescu/UBnormal)]

<p align = "justify"> 
Th dataset is a novel benchmark created for the purpose of supervised open-set video anomaly detection. UBnormal is composed of multiple virtual scenes generated using Cinema4D software with virtual animated characters and objects placed in real-world backgrounds. It contains a variety of normal and abnormal events across different scenes.The dataset includes a range of normal activities such as walking, talking on the phone, and standing, as well as 22 types of abnormal events like running, falling, fighting, sleeping, and car crashes. The abnormal events in the training set are distinct from those in the test set, aligning with the open-set condition.Unlike existing datasets, UBnormal provides pixel-level annotations for abnormal events during training. This allows for the use of fully-supervised learning methods.The dataset features 29 different natural images representing various environments like street scenes, train stations, and office rooms. Each background image is used to create a virtual 3D scene, generating an average of 19 videos per scene.UBnormal includes multiple object categories such as people, cars, skateboards, bicycles, and motorcycles, which can perform both normal and abnormal actions. The dataset employs 19 different characters to animate the videos, with variations in clothing colors and hair color to increase diversity.
</p>
<p align = "center">
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%8712.1.png" width="675" />
<img src="https://github.com/kanyutingfeng/survey-dataset-photos/blob/photos/%E5%9B%BE%E7%89%8712.2.png" width="225" />
</p>

## 1.5 Video Anomaly Caption/Retrieval

### 1.5.1 UCA:Towards Surveillance Video-and-Language Understanding: New Dataset Baselines and Challenges(CVPR 2024)

[[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Yuan_Towards_Surveillance_Video-and-Language_Understanding_New_Dataset_Baselines_and_Challenges_CVPR_2024_paper.pdf)][[Homepage](https://xuange923.github.io/Surveillance-Video-Understanding)]

<p align = "justify">
The UCA (UCF-Crime Annotation) dataset is a pioneering multimodal surveillance video dataset. The dataset is created by manually annotating the real-world surveillance dataset UCFCrime with fine-grained event content and timing. It is designed to support research in surveillance video-and-language understanding (VALU).The dataset contains 23,542 sentences with an average length of 20 words. These annotations are applied to videos totaling 110.7 hours in length.The annotation process involved filtering out low-quality videos from the original UCF-Crime dataset, followed by fine-grained language annotation that recorded each event in detail with time stamps.UCA is intended to facilitate research in intelligent public security, particularly in tasks related to multimodal surveillance video comprehension, such as retrieving detailed event queries with temporal information, captioning surveillance videos, and multimodal anomaly detection.
</p>

### 1.5.2 VAD-Instruct50k:Holmes-VAD: Towards Unbiased and Explainable Video Anomaly Detection via Multi-modal LLM

[[Paper](https://arxiv.org/pdf/2406.12235)][[Homepage](https://holmesvad.github.io/)]

<p align = "justify">
The dataset is a large-scale multimodal video anomaly detection benchmark.The dataset aims to provide precise temporal supervision and rich multimodal instructions to enable accurate anomaly localization and comprehensive explanations in video anomaly detection.VAD-Instruct50k is created using a semi-automatic labeling paradigm. This method involves efficient single-frame annotations applied to untrimmed videos, which are then synthesized into high-quality analyses of both abnormal and normal video clips.The videos for the dataset are primarily gathered from open-source datasets, including a large number of untrimmed videos with video-level anomaly labels.The collected videos are enhanced by generating reliable video event clips around the single-frame annotated frames and providing textual descriptions through human effort or foundation models. The dataset includes single-frame temporal annotations and explanatory text descriptions for both untrimmed videos and trimmed abnormal/normal video clips.
</p>

<a id="ucfcrimear-xdviolencear"></a>
### 1.5.3&1.5.4 UCFCrimeAR & XDViolenceAR:Toward Video Anomaly Retrieval From Video Anomaly Detection: New Benchmarks and Model(TIP 2024)

[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10471334)][[Homepage](https://github.com/Roc-Ng/VAR?tab=readme-ov-file)]

<p align = "justify">
The UCFCrime-AR and XDViolence-AR datasets are two large-scale benchmarks created for the purpose of facilitating research in the field of video anomaly analysis, specifically for the novel task of Video Anomaly Retrieval (VAR).
</p>
<p align = "justify">
UCFCrime-AR is constructed from the UCFCrime dataset.Videos are annotated with text descriptions in both Chinese and English by experienced annotators. Annotators focus on describing the anomalous events in detail to capture the fine differences between similar videos. And it is designed for video-text VAR, allowing the retrieval of videos based on text descriptions of anomalous events.
</p>
<p align = "justify">
XDViolence-AR is based on the XD-Violence dataset, another comprehensive VAD dataset. The dataset focuses on the audio-visual aspect of VAR due to the complexity of video content in this dataset. Unlike UCFCrime-AR, which uses text descriptions, XDViolence-AR utilizes synchronous audios for cross-modal anomaly retrieval, capitalizing on the natural audio-visual information present in the videos, with the aim of retrieving videos based on audio queries.
Both UCFCrime-AR and XDViolence-AR have significantly longer average video lengths compared to traditional video retrieval datasets, emphasizing the goal of VAR to retrieve long and untrimmed videos, which aligns with realistic requirements and poses a more challenging task.
</p>
<p align = "justify">
These benchmarks are designed to be used in cross-modal retrieval scenarios, where the system is required to retrieve videos based on either text descriptions (for UCFCrime-AR) or audio (for XDViolence-AR).
</p>

### 1.5.5 UCCD:Human-centric Behavior Description in Videos: New Benchmark and Model(TMM 2024)

[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10582309v)][[Homepage](https://github.com/lingruzhou/UCCD)]

<p align = "justify">
The dataset is a human-centric video surveillance captioning dataset. This dataset is designed to address the challenge of describing the behavior of each individual within a video, especially in complex scenarios with multiple individuals.The dataset provides detailed descriptions of the dynamic behaviors of individuals, allowing for a more nuanced understanding of situations in video surveillance.The dataset contains comprehensive annotations for each individual, including their location, clothing, and interactions with other elements in the scene. For each person, bounding boxes are provided for the first frame they appear, along with time stamps of their appearance and disappearance.UCCD differentiates itself by offering instance-level descriptions of individual behaviors within a video, segmenting the video based on individuals rather than events.
</p>

# 2. Performance Evaluation

## 2.1 frame-level metrics

### 2.1.1 AUC (Area Under the Curve)[[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.pdf)][[Frame-level AUC](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhong_Graph_Convolutional_Label_Noise_Cleaner_Train_a_Plug-And-Play_Action_Classifier_CVPR_2019_paper.pdf)][[Pixel-level AUC](https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Lu_Abnormal_Event_Detection_2013_ICCV_paper.pdf)]

<p align = "justify"> 
AUC refers to the area under the Receiver Operating Characteristic (ROC) Curve. The ROC Curve is created by plotting the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Large%20TPR=\frac{TP}{TP+FN}" alt="TPR"/>
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Large%20FPR=\frac{FP}{FP+TN}" alt="FPR"/>
</p>


<p align = "justify"> 
Where TP(True Positives) is the number of actual positive instances, i.e., the count of samples correctly predicted as positive. TN(True Negatives) is the number of actual negative instances, i.e., the count of samples correctly predicted as negative. FP(False Positives) is the number of false positive instances, i.e., the count of samples incorrectly predicted as positive. FN(False Negatives) is the number of false negative instances, i.e., the count of samples incorrectly predicted as negative.
</p>

<p align = "justify"> 
AUC is used to measure the overall performance of a classifier, especially useful when dealing with class imbalance. It provides a more robust measure of performance than accuracy alone. The value of AUC ranges from 0 to 1, with higher values indicating better model performance.
</p>
<p align = "justify"> 
Frame-level AUC pays special attention to the detection of the video frame level, that is the classification accuracy of the model on whether each frame in the video contains abnormal events.
</p>
<p align = "justify"> 
Pixel-level AUC is a more refined evaluation metric that evaluates the performance of the model at the pixel level. This means that the model should not only detect abnormal frames, but also be able to locate specific areas in the frame where the abnormality occurs. This evaluation method puts higher requirements on the spatial positioning ability of the model.
</p>

### 2.1.2 AP (Average Precision)[[Paper](https://link.springer.com/chapter/10.1007/978-3-030-58577-8_20)][[mAP](https://link.springer.com/article/10.1007/s11042-014-1984-4)]

<p align = "justify"> 
AP stands for the area under the Precision-Recall Curve. Precision is the proportion of positive identifications that were actually correct, while recall (or sensitivity) is the proportion of actual positive cases that were identified correctly. AP is particularly useful in situations where the number of positive samples (such as anomalies) is low, which is common in class-imbalanced datasets. Like AUC, a higher AP value indicates better performance, as it balances precision and recall.
</p>

<p align = "justify"> 
mAP is the mean of the Average Precision scores across different classes. In multi-class classification tasks, each class has its own Precision-Recall Curve, and mAP calculates the average of the AP for each class, providing an overall performance measure of the model across all classes. mAP is a widely used metric for multi-class detection tasks, especially in object detection. It provides a comprehensive measure of the model's ability to identify and localize objects across different categories.
</p>

### 2.1.3 EER (Equal Error Rate) and EDR (Equal Detected Rate)[[Paper](https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Lu_Abnormal_Event_Detection_2013_ICCV_paper.pdf)]

<p align = "justify"> 
EER refers to the error rate where the false positive rate (FPR) and the false negative rate (FNR) are equal on the Receiver Operating Characteristic (ROC) curve. At this point, the detection system has an equal number of true positives (TP) and false negatives (FN), meaning the number of correctly detected anomalies is the same as the number of missed anomalies. EER is a crucial performance measure because it provides a balance point where the sensitivity and specificity of the detection system are equally weighted.
</p>

<p align = "justify"> 
EDR typically refers to the proportion of total anomalies that are detected by the system at a specific detection threshold. This metric focuses on the recall of the detection system, which is the ratio of the number of correctly detected anomalies to the total number of actual anomalies.
</p>

<p align = "justify"> 
EER provides a point of balance, while EDR emphasizes the completeness of detection, especially in anomaly detection where a high recall rate is often more critical, as missing a true anomaly can have more severe consequences than falsely flagging a normal event as an anomaly.
</p>

### 2.1.4 Accuracy[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6239234)]

<p align = "justify"> 
Accuracy is a performance measurement for classification models or diagnostic tests that reflects the ratio of the number of correct predictions to the total number of predictions. It is one of the most intuitive performance metrics, especially in binary or multi-class classification problems. Accuracy is typically calculated using the following formula:  
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Large%20Accuracy=\frac{TP+TN}{TP+TN+FP+FN}" alt="Accuracy"/>
</p>

<p align = "justify"> 
Where TP(True Positives) is the number of actual positive instances, i.e., the count of samples correctly predicted as positive. TN(True Negatives) is the number of actual negative instances, i.e., the count of samples correctly predicted as negative.FP(False Positives) is the number of false positive instances, i.e., the count of samples incorrectly predicted as positive. FN(False Negatives) is the number of false negative instances, i.e., the count of samples incorrectly predicted as negative.
</p>
<p align = "justify"> 
Accuracy provides a straightforward metric to assess the overall performance of a model, indicating the probability that the model makes correct predictions. Accuracy is easy to understand and calculate, offering a quick overview of model performance. But in cases of class imbalance, accuracy can be misleading. For instance, if the majority of samples belong to one class, a model might achieve high accuracy by simply predicting that class for all samples, even if it is inaccurate for the minority class.
</p>
<p align = "justify"> 
Accuracy is often used for a quick assessment of model performance but is usually complemented by other metrics such as precision, recall, and the F1 score to fully evaluate model performance, especially with imbalanced datasets. When using accuracy as an evaluation metric, it is important to consider the distribution of the dataset. If the dataset has an uneven distribution of positive and negative samples, other metrics may need to be used in conjunction with accuracy to provide a more comprehensive assessment of the model's performance.
</p>

## 2.2 pixel-level metrics

### 2.2.1 TIoU (Temporal Intersection over Union)[[Paper](https://dl.acm.org/doi/pdf/10.1145/3343031.3350998)]

<p align = "justify"> 
TIoU is a video anomaly detection-specific metric that combines the temporal anomaly detection score with the model's ability to spatially locate the learned pattern of anomalies. TIoU evaluates the model's spatial localization accuracy by calculating the intersection over union (IoU) between the model's predicted anomaly region and the manually annotated region. This metric not only considers the model's ability to detect anomalies over time but also assesses the spatial accuracy of the model's anomaly localization.
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Large%20TIoU=\frac{1}{M}\sum_{j=1}^{M}%20\frac{Area_p%20\cap%20Area_g}{Area_p%20\cup%20Area_g}%20\cdot%20\mathbb{I}[P_j%20\geq%20Threshold]" alt="TIoU"/>
</p>


<p align="justify">
where the indicator 
<img src="https://latex.codecogs.com/svg.image?\inline%20\mathbb{I}[\cdot]%20\in%20\{0,1\}" alt="indicator"/> 
indicates whether the given 
<img src="https://latex.codecogs.com/svg.image?\inline%20j^{th}" alt="jth"/> 
anomaly clip are predicted as anomaly according to the probability score 
<img src="https://latex.codecogs.com/svg.image?\inline%20P_j" alt="Pj"/>, 
<img src="https://latex.codecogs.com/svg.image?\inline%20\text{Area}_p" alt="Areap"/> 
represents the area of predicted anomalous region, 
<img src="https://latex.codecogs.com/svg.image?\inline%20\text{Area}_g" alt="Areag"/> 
is the area of annotated region, and 
<img src="https://latex.codecogs.com/svg.image?\inline%20M" alt="M"/> 
is the number of clips that anomaly occurs.
</p>


### 2.2.2 RBDR (Region-based Detection Rate) and TBDR (Track-based Detection Rate)[[Paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Ramachandra_Street_Scene_A_new_dataset_and_evaluation_protocol_for_video_WACV_2020_paper.pdf)]

<p align = "justify"> 
The RBDR evaluates the model's ability to accurately localize the spatial extent of anomalies within video frames. It is particularly useful for assessing how well a model can identify the regions where anomalies occur. This metric compares the detected anomaly regions with the ground truth annotations to compute a score. The comparison is often done using the Intersection over Union (IoU), which measures the overlap between the predicted region and the actual region of the anomaly. A higher RBDR score indicates better spatial localization performance, meaning the model is more accurate in identifying the correct area of the video frame where the anomaly happens.
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Large%20RBDR=%20\frac{\text{num.%20of%20anomalous%20regions%20detected}}{\text{total%20num.%20of%20anomalous%20regions}}" alt="RBDR"/>
</p>


<p align = "justify"> 
The TBDR is focused on the model's capability to detect and track anomalies over time, providing a measure of how well the model can localize anomalies across consecutive video frames. This criterion is especially relevant for scenarios where anomalies have a temporal component, such as an object moving in an unusual way or an event unfolding over several frames. Similar to RBDC, TBDR also uses IoU to measure the overlap between the predicted anomaly track and the ground truth track. However, it considers the temporal continuity, ensuring that the model not only detects the anomaly in individual frames but also maintains the correct tracking of the anomaly across the video sequence.
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Large%20TBDR=%20\frac{\text{num.%20of%20anomalous%20tracks%20detected}}{\text{total%20num.%20of%20anomalous%20tracks}}" alt="TBDR"/>
</p>


<p align = "justify"> 
While AUC metrics provide an overall performance measure, RBDR and TBDR offer insights into the spatial and temporal accuracy of the model's anomaly localization capabilities.
</p>


## 2.3 sentence-level and reasoning-based metrics

### 2.3.1 BLUE(Bilingual Evaluation Understudy)[[Paper](https://aclanthology.org/P02-1040.Pdf)]
### BLEU Evaluation Metric

### BLEU Score Overview

<p align = "justify"> 
BLEU is a metric for evaluating the quality of machine translation. It measures the accuracy of a translation by comparing the n-gram overlap between the machine-generated translation and one or more human reference translations, while incorporating a length penalty to prevent overly short outputs. BLEU is primarily designed for **corpus-level evaluation** and is less reliable for single-sentence scoring. The BLEU score ranges from 0 to 1; even high-quality human translations rarely achieve a score close to 1. Using more reference translations generally leads to higher BLEU scores. BLEU is better suited for evaluating the overall performance of large-scale translation systems rather than individual sentences.
</p>

#### a. Modified n-gram Precision
For n-grams of order n, the modified precision is calculated as:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Large%20p_n=%20\frac{\sum_{C%20\in%20\text{Candidates}}%20\sum_{n\text{-gram}%20\in%20C}%20\text{Count}_{\text{clip}}(n\text{-gram})}{\sum_{C'%20\in%20\text{Candidates}}%20\sum_{n\text{-gram}%20\in%20C'}%20\text{Count}(n\text{-gram})}" alt="p_n公式"/>
</p>

 Here, ![Count](https://latex.codecogs.com/svg.image?\inline%20\text{Count}_{\text{clip}}(n\text{-gram})) is the clipped count, i.e., the maximum number of times an n-gram appears in any single reference translation.
 Typically, precisions are computed for 1-gram up to 4-gram.

#### b. Brevity Penalty (BP)
To avoid rewarding translations that are too short, BLEU applies a brevity penalty:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Large%20\text{BP}=%20\begin{cases}1,%20&%20\text{if}\;%20c>r%20\\%20e^{(1-r/c)},%20&%20\text{if}\;%20c\leq%20r\end{cases}" alt="BP"/>
</p>


 ![c](https://latex.codecogs.com/svg.image?\inline%20c) is the total length of the candidate translations.
 ![r](https://latex.codecogs.com/svg.image?\inline%20r) is the effective reference corpus length (best matching length).

#### c. Final BLEU Score

The overall BLEU score combines the modified n-gram precisions with the brevity penalty:


<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Large%20\text{BLEU}=%20\text{BP}%20\cdot%20\exp\left(%20\sum_{n=1}^{N}%20w_n%20\log%20p_n%20\right)" alt="BLEU公式"/>
</p>


 Usually, ![N=4](https://latex.codecogs.com/svg.image?\inline%20N=4) and weights ![w_n = 1/4](https://latex.codecogs.com/svg.image?\inline%20w_n=%20\frac{1}{4}).
 This formula represents the geometric mean of the modified n-gram precisions multiplied by the brevity penalty.

### 2.3.2 CIDEr(Consensus-based Image Description Evaluation)[[Paper](https://openaccess.thecvf.com/content_cvpr_2015/papers/Vedantam_CIDEr_Consensus-Based_Image_2015_CVPR_paper.pdf)]
## CIDEr: Consensus-based Image Description Evaluation

CIDEr is a metric specifically designed for image captioning tasks. It measures the similarity between a machine-generated caption and the **consensus of multiple human-written captions** for the same image. Unlike evaluating grammar or the “best” description, CIDEr focuses on how **human-like** a caption is by comparing it to several reference captions. CIDEr uses **TF-IDF weighted n-gram cosine similarity** to emphasize visually informative words while suppressing common ones. By combining n-grams of lengths 1 to 4, it captures both syntactic and semantic information, providing a more accurate assessment of caption relevance.

#### a. TF-IDF Weight for n-gram \omega_k

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Large%20g_k(s_{ij})=%20\frac{h_k(s_{ij})}{\sum_{\omega_l%20\in%20\Omega}%20h_l(s_{ij})}%20\cdot%20\log\!\left(\frac{|I|}{\sum_{p%20\in%20I}%20\min(1,%20\sum_q%20h_k(s_{pq}))}\right)" alt="g_k"/>
</p>


![h_k(s_{ij})](https://latex.codecogs.com/svg.image?\inline%20h_k(s_{ij})): frequency of n-gram ![\omega_k](https://latex.codecogs.com/svg.image?\inline%20\omega_k) in sentence ![s_{ij}](https://latex.codecogs.com/svg.image?\inline%20s_{ij}) (Term Frequency)  
![|I|](https://latex.codecogs.com/svg.image?\inline%20|I|): total number of images  
Denominator: number of images containing n-gram ![\omega_k](https://latex.codecogs.com/svg.image?\inline%20\omega_k) (Inverse Document Frequency)



#### b. CIDEr Score for n-grams of length 

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Large%20\text{CIDEr}_n(c_i,%20S_i)=%20\frac{1}{m}%20\sum_{j=1}^{m}%20\frac{\mathbf{g}_n(c_i)\cdot\mathbf{g}_n(s_{ij})}{\|\mathbf{g}_n(c_i)\|\cdot\|\mathbf{g}_n(s_{ij})\|}" alt="CIDEr"/>
</p>

 ![c_i](https://latex.codecogs.com/svg.image?\inline%20c_i): candidate caption
 ![S_i](https://latex.codecogs.com/svg.image?\inline%20S_i%20=%20\{s_{i1},%20...,%20s_{im}\}): set of reference captions
 Computes the average cosine similarity between the TF-IDF vectors of the candidate and each reference caption



#### c. Final CIDEr Score (combining 1- to 4-grams)

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Large%20\text{CIDEr}(c_i,%20S_i)=%20\sum_{n=1}^{N}%20w_n%20\cdot%20\text{CIDEr}_n(c_i,%20S_i)" alt="CIDEr"/>
</p>

 Typically ![N=4](https://latex.codecogs.com/svg.image?\inline%20N=4) with uniform weights ![w_n = 1/4](https://latex.codecogs.com/svg.image?\inline%20w_n=%20\frac{1}{4})
 

### 2.3.3 METEOR(Metric for Evaluation of Translation with Explicit ORdering)[[Paper](https://aclanthology.org/W05-0909.pdf)]
### METEOR: An Automatic Metric for MT Evaluation

METEOR is an automatic metric for machine translation (MT) evaluation. It was designed to address several limitations of BLEU, aiming for better correlation with human judgments by using flexible word matching, recall-oriented scoring, and explicit handling of word order. METEOR is an automatic machine translation evaluation metric that improves upon BLEU by incorporating flexible word matching (exact, stemmed, and synonym matches), placing greater emphasis on recall than precision, and penalizing word order disruptions through a fragmentation penalty. It aligns translations using a multi-stage process to minimize crossing links, computes a weighted harmonic mean of precision and recall, and applies penalties for disordered matches. Overall, METEOR provides sentence-level scores with stronger correlation to human judgments and handles linguistic variations more effectively than BLEU. METEOR produces scores ranging from 0 to 1, where higher values indicate better translation quality. For each sentence, the best score among multiple reference translations is selected, and an aggregate score is calculated at the corpus level to evaluate overall system performance. METEOR is a robust, recall-oriented machine translation evaluation metric. By incorporating stemming, synonymy, and word order sensitivity, it achieves superior correlation with human assessments compared to BLEU.

#### a. Word Alignment
System and reference translations are aligned in stages: Exact match; Stemmed match (Porter Stemmer); Synonym match (WordNet). Each stage aligns remaining unmapped words, minimizing crossing links to preserve word order.


#### b. Precision & Recall Calculation

Let:
 ![m](https://latex.codecogs.com/svg.image?\inline%20m) : number of mapped unigrams
 ![t](https://latex.codecogs.com/svg.image?\inline%20t) : total unigrams in system translation
 ![r](https://latex.codecogs.com/svg.image?\inline%20r) : total unigrams in reference translation

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Large%20P=%20\frac{m}{t},%20\quad%20R=%20\frac{m}{r}" alt="Precision and Recall"/>
</p>



#### c. Weighted Harmonic Mean (Fmean)


<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Large%20\text{Fmean}=%20\frac{10PR}{R+9P}" alt="Fmean公式"/>
</p>

*(Recall is weighted 9× over precision)*


#### d. Fragmentation Penalty

Let:
 chunks: minimum number of contiguous matched segments

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Large%20\text{Penalty}=%200.5%20\times%20\left(%20\frac{\text{chunks}}{m}%20\right)^3" alt="Penalty"/>
</p>



#### e. Final METEOR Score

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Large%20\text{METEOR%20Score}=%20\text{Fmean}%20\times%20(1-%20\text{Penalty})" alt="METEOR"/>
</p>


### 2.3.4 ROUGE(Recall-Oriented Understudy for Gisting Evaluation)[[Paper](https://aclanthology.org/W04-1013.pdf)]

ROUGE is a family of automatic metrics used to evaluate the quality of a **candidate summary** by measuring overlapping units—such as n-grams, longest common subsequences, and skip-bigrams—against one or more **human reference summaries**. It focuses primarily on recall but can incorporate precision and F-measures depending on the variant. For practical use, ROUGE-2, ROUGE-L, ROUGE-W, and ROUGE-S are recommended for single-document summaries, while ROUGE-1, ROUGE-L, ROUGE-SU4, and ROUGE-SU9 work best for headlines or very short summaries. For multi-document summaries, using ROUGE-1, ROUGE-2, ROUGE-S4/S9, and ROUGE-SU4/S9 combined with stop-word removal provides more stable correlation with human judgments. Employing multiple reference summaries consistently improves evaluation reliability. Additionally, stemming slightly enhances results, and stop-word removal is especially beneficial for multi-document tasks. Empirical evaluations on DUC datasets show that ROUGE achieves Pearson correlations up to 0.88 for single-document 100-word summaries, up to 0.97 for very short 10-word summaries, and between 0.70 and 0.85 for multi-document 100-word summaries. In summary, a good rule of thumb is to use ROUGE-2 for long single-document summaries, ROUGE-1/L/SU4 for headlines, and ROUGE-SU4 with stop-word removal for multi-document summaries, always leveraging multiple reference summaries when available.


#### ROUGE Variants & Core Formulas

| Variant | Unit Measured | Intuition | Formula (Single Reference) |
|----------|---------------|------------|-----------------------------|
| **ROUGE-N** | n-gram | n-gram recall vs. reference | ![ROUGE-N](https://latex.codecogs.com/svg.image?\text{ROUGE-N}=%20\frac{\sum_{S%20\in%20\{\text{ref}\}}%20\sum_{\text{gram}_n}%20\text{Count}_{\text{match}}(\text{gram}_n)}{\sum_{S%20\in%20\{\text{ref}\}}%20\sum_{\text{gram}_n}%20\text{Count}(\text{gram}_n)}) |
| **ROUGE-L** | Longest Common Subsequence (LCS) | Sentence-level in-order word overlap | ![ROUGE-L](https://latex.codecogs.com/svg.image?\text{ROUGE-L}=F_{\text{lcs}}=%20\frac{(1+\beta^2)R_{\text{lcs}}P_{\text{lcs}}}{R_{\text{lcs}}+\beta^2P_{\text{lcs}}},%20\quad%20\text{where}%20R_{\text{lcs}}=%20\frac{\text{LCS}(X,Y)}{m},%20P_{\text{lcs}}=%20\frac{\text{LCS}(X,Y)}{n}) |
| **ROUGE-W** | Weighted LCS | Rewards **consecutive** matches | ![ROUGE-W](https://latex.codecogs.com/svg.image?\text{ROUGE-W}=F_{\text{wlcs}},%20\text{with%20weighted%20score}%20\text{WLCS}(X,Y)%20\text{using}%20f(k)=k^2) |
| **ROUGE-S** | Skip-bigram | Any ordered word-pair (within skip-gap) | ![ROUGE-S](https://latex.codecogs.com/svg.image?\text{ROUGE-S}=F_{\text{skip2}}=%20\frac{(1+\beta^2)R_{\text{skip2}}P_{\text{skip2}}}{R_{\text{skip2}}+\beta^2P_{\text{skip2}}}) |
| **ROUGE-SU** | Skip-bigram + unigram | Adds unigram to handle zero-match cases | Same as ROUGE-S but numerator and denominator include unigram hits |



> In DUC evaluations, ![beta→∞](https://latex.codecogs.com/svg.image?\inline%20\beta%20\to%20\infty) is used, so only **recall** is reported for simplicity.


#### Multi-Reference Handling

For multiple reference summaries ![{r1, r2, ..., rM}](https://latex.codecogs.com/svg.image?\inline%20\{r_1,%20r_2,%20...,%20r_M\}), compute ROUGE scores between the candidate ![c](https://latex.codecogs.com/svg.image?\inline%20c) and each reference, then take the maximum:


<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Large%20\text{ROUGE-N}_{\text{multi}}=%20\max_{i=1..M}%20\text{ROUGE-N}(r_i,%20c)" alt="ROUGE-N_multi"/>
</p>


Official implementations use jackknifing: averaging max scores over leave-one-out reference sets to stabilize comparisons.

#### Key Properties

| Variant  | Requires Consecutive Matches? | Captures Word Order? | Sensitive to Sentence Structure? |
|-----------|-------------------------------|----------------------|----------------------------------|
| **ROUGE-N** | Yes (for ![n≥2](https://latex.codecogs.com/svg.image?\inline%20n%20\ge%202)) | Partial | Low |
| **ROUGE-L** | No (in-sequence only) | Yes | High |
| **ROUGE-W** | No (but boosts consecutive) | Yes | High |
| **ROUGE-S** | No | Yes (via skip-bigrams) | Medium |
| **ROUGE-SU** | No | Yes (+ unigram) | Medium |



### 2.3.5 MMEval()[[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Du_Uncovering_What_Why_and_How_A_Comprehensive_Benchmark_for_Causation_CVPR_2024_paper.pdf)]

Traditional Natural Language Generation (NLG) metrics like BLEU and ROUGE are text-only and fail to consider the visual evidence essential in video-text tasks. Causation Understanding of Video Anomaly requires evaluating free-text explanations for **cause** and **effect** grounded in video content. To address this, **MMEval** is proposed as a **multimodal** evaluation metric that leverages both video and text to align closely with human judgment. It works by constructing task-specific natural language prompts, selecting key video frames based on an importance curve that highlights anomalous segments, and feeding these frames along with the prompt and candidate answer into a frozen Video-ChatGPT model. The model outputs a scalar score (0–100) and a short rationale, enabling transparent and explainable evaluation. MMEval is task-agnostic (using the same model with different prompts) and achieves high human consistency with Spearman correlation between 0.82 and 0.89. The approach reduces noise and computation by densely sampling only the most relevant frames according to a threshold set on the importance curve.

Given a video clip ![V](https://latex.codecogs.com/svg.image?\inline%20V), a task-specific prompt ![P_task](https://latex.codecogs.com/svg.image?\inline%20P_{\text{task}}) (for Description, Cause, or Effect), and a candidate free-text answer ![A](https://latex.codecogs.com/svg.image?\inline%20A), MMEval uses a frozen Video-ChatGPT model ![\Phi(\cdot)](https://latex.codecogs.com/svg.image?\inline%20\Phi(\cdot)) to compute a scalar score and ranking:


<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Large%20s=%20\Phi(V',%20P_{\text{task}},%20A)%20\in%20[0,%20100]" alt="s"/>
</p>


<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Large%20\text{rank}(A)=%20\text{ordinal%20position%20after%20sorting%20all%20}%20s%20\text{%20in%20descending%20order}" alt="rank(A)"/>
</p>


Here, ![V](https://latex.codecogs.com/svg.image?\inline%20V) is a subset of video frames selected by thresholding the importance curve ![I(t)](https://latex.codecogs.com/svg.image?\inline%20I(t)) (which measures anomaly relevance over time):

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Large%20\tau=%20\mu+%20\sigma" alt="tau公式"/>
</p>

Frames with importance scores ![I(t)≥τ](https://latex.codecogs.com/svg.image?\inline%20I(t)%20\geq%20\tau) are densely sampled (at 10 fps) to focus the evaluation on key segments, reducing noise and computational cost.

















































