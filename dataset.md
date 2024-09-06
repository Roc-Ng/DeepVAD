# 1. Dataset

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

## 1.5 Video Anomaly Caption/Retrieval

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

### 2.1 AUC (Area Under the Curve)[[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.pdf)][[Frame-level AUC](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhong_Graph_Convolutional_Label_Noise_Cleaner_Train_a_Plug-And-Play_Action_Classifier_CVPR_2019_paper.pdf)][[Pixel-level AUC](https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Lu_Abnormal_Event_Detection_2013_ICCV_paper.pdf)]

<p align = "justify"> 
AUC refers to the area under the Receiver Operating Characteristic (ROC) Curve. The ROC Curve is created by plotting the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.
</p>

$$ TPR=\frac{TP}{TP+FN} $$

$$ FPR=\frac{FP}{FP+TN} $$

<p align = "justify"> 
Where: TP(True Positives) is the number of actual positive instances, i.e., the count of samples correctly predicted as positive.TN(True Negatives) is the number of actual negative instances, i.e., the count of samples correctly predicted as negative.FP(False Positives) is the number of false positive instances, i.e., the count of samples incorrectly predicted as positive.FN(False Negatives) is the number of false negative instances, i.e., the count of samples incorrectly predicted as negative.
</p>

<p align = "justify"> 
AUC is used to measure the overall performance of a classifier, especially useful when dealing with class imbalance. It provides a more robust measure of performance than accuracy alone. The value of AUC ranges from 0 to 1, with higher values indicating better model performance.
</p>
<p align = "justify"> 
Frame-level AUC pays special attention to the detection of the video frame level, that is, the classification accuracy of the model on whether each frame in the video contains abnormal events.
</p>
<p align = "justify"> 
Pixel-level AUC is a more refined evaluation metric that evaluates the performance of the model at the pixel level. This means that the model should not only detect abnormal frames, but also be able to locate specific areas in the frame where the abnormality occurs. This evaluation method puts higher requirements on the spatial positioning ability of the model.
</p>

### 2.2 AP (Average Precision)[[Paper](https://link.springer.com/chapter/10.1007/978-3-030-58577-8_20)][[mAP](https://link.springer.com/article/10.1007/s11042-014-1984-4)]

<p align = "justify"> 
AP stands for the area under the Precision-Recall Curve. Precision is the proportion of positive identifications that were actually correct, while recall (or sensitivity) is the proportion of actual positive cases that were identified correctly. AP is particularly useful in situations where the number of positive samples (such as anomalies) is low, which is common in class-imbalanced datasets. Like AUC, a higher AP value indicates better performance, as it balances precision and recall.
</p>

<p align = "justify"> 
mAP is the mean of the Average Precision scores across different classes. In multi-class classification tasks, each class has its own Precision-Recall Curve, and mAP calculates the average of the AP for each class, providing an overall performance measure of the model across all classes.mAP is a widely used metric for multi-class detection tasks, especially in object detection. It provides a comprehensive measure of the model's ability to identify and localize objects across different categories.
</p>

### 2.3 RBDC (Region-based Detection Criterion) and TBDC (Track-based Detection Criterion)[[Paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Ramachandra_Street_Scene_A_new_dataset_and_evaluation_protocol_for_video_WACV_2020_paper.pdf)]

<p align = "justify"> 
The RBDC evaluates the model's ability to accurately localize the spatial extent of anomalies within video frames. It is particularly useful for assessing how well a model can identify the regions where anomalies occur.This metric compares the detected anomaly regions with the ground truth annotations to compute a score. The comparison is often done using the Intersection over Union (IoU), which measures the overlap between the predicted region and the actual region of the anomaly.A higher RBDC score indicates better spatial localization performance, meaning the model is more accurate in identifying the correct area of the video frame where the anomaly happens.
</p>

$$ RBDC = \frac{num.\ of\ anomalous\ regions\ detected} {total\ num.\ of\ anomalous\ regions} $$

<p align = "justify"> 
The TBDC is focused on the model's capability to detect and track anomalies over time, providing a measure of how well the model can localize anomalies across consecutive video frames. This criterion is especially relevant for scenarios where anomalies have a temporal component, such as an object moving in an unusual way or an event unfolding over several frames. Similar to RBDC, TBDC also uses IoU to measure the overlap between the predicted anomaly track and the ground truth track. However, it considers the temporal continuity, ensuring that the model not only detects the anomaly in individual frames but also maintains the correct tracking of the anomaly across the video sequence.
</p>

$$ TBDC = \frac{num.\ of\ anomalous\ tracks\ detected} {total\ num.\ of\ anomalous\ tracks} $$

<p align = "justify"> 
While AUC metrics provide an overall performance measure, RBDC and TBDC offer insights into the spatial and temporal accuracy of the model's anomaly localization capabilities.
</p>

### 2.4 EER (Equal Error Rate) and EDR (Equal Detected Rate)[[Paper](https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Lu_Abnormal_Event_Detection_2013_ICCV_paper.pdf)]

<p align = "justify"> 
EER refers to the error rate where the false positive rate (FPR) and the false negative rate (FNR) are equal on the Receiver Operating Characteristic (ROC) curve.At this point, the detection system has an equal number of true positives (TP) and false negatives (FN), meaning the number of correctly detected anomalies is the same as the number of missed anomalies.EER is a crucial performance measure because it provides a balance point where the sensitivity and specificity of the detection system are equally weighted.
</p>

<p align = "justify"> 
EDR typically refers to the proportion of total anomalies that are detected by the system at a specific detection threshold.This metric focuses on the recall of the detection system, which is the ratio of the number of correctly detected anomalies to the total number of actual anomalies.
</p>

<p align = "justify"> 
EER provides a point of balance, while EDR emphasizes the completeness of detection, especially in anomaly detection where a high recall rate is often more critical, as missing a true anomaly can have more severe consequences than falsely flagging a normal event as an anomaly.
</p>

### 2.5 Accuracy[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6239234)]

<p align = "justify"> 
Accuracy is a performance measurement for classification models or diagnostic tests that reflects the ratio of the number of correct predictions to the total number of predictions. It is one of the most intuitive performance metrics, especially in binary or multi-class classification problems.Accuracy is typically calculated using the following formula:  
</p>

$$ Accuracy=\frac{TP+TN}{TP+TN+FP+FN} $$ 

<p align = "justify"> 
Where: TP(True Positives) is the number of actual positive instances, i.e., the count of samples correctly predicted as positive.TN(True Negatives) is the number of actual negative instances, i.e., the count of samples correctly predicted as negative.FP(False Positives) is the number of false positive instances, i.e., the count of samples incorrectly predicted as positive.FN(False Negatives) is the number of false negative instances, i.e., the count of samples incorrectly predicted as negative.
</p>
<p align = "justify"> 
Accuracy provides a straightforward metric to assess the overall performance of a model, indicating the probability that the model makes correct predictions.Accuracy is easy to understand and calculate, offering a quick overview of model performance.But in cases of class imbalance, accuracy can be misleading. For instance, if the majority of samples belong to one class, a model might achieve high accuracy by simply predicting that class for all samples, even if it is inaccurate for the minority class.
</p>
<p align = "justify"> 
Accuracy is often used for a quick assessment of model performance but is usually complemented by other metrics such as precision, recall, and the F1 score to fully evaluate model performance, especially with imbalanced datasets.When using accuracy as an evaluation metric, it is important to consider the distribution of the dataset. If the dataset has an uneven distribution of positive and negative samples, other metrics may need to be used in conjunction with accuracy to provide a more comprehensive assessment of the model's performance.
</p>

### 2.6TIoU(Temporal Intersection over Union)[[Paper](https://dl.acm.org/doi/pdf/10.1145/3343031.3350998)]

<p align = "justify"> 
TIoU is a video anomaly detection-specific metric that combines the temporal anomaly detection score with the model's ability to spatially locate the learned pattern of anomalies.TIoU evaluates the model's spatial localization accuracy by calculating the intersection over union (IoU) between the model's predicted anomaly region and the manually annotated region. This metric not only considers the model's ability to detect anomalies over time but also assesses the spatial accuracy of the model's anomaly localization.
</p>

$$ TIoU=\frac{1}{M}\sum_{j=1} ^M \frac{Area_p \cap Area_g}{Area_p \cup Area_g} \cdot II[P_j \geq Threshold] $$

<p align = "justify"> 
where the indicator II[.] ∈ {0,1} indicates whether the given 𝑗𝑡ℎ anomaly clip are predicted as anomaly according to the probability score 𝑃𝑗, 𝐴𝑟𝑒𝑎_𝑝represents the area of predicted anomalous region, 𝐴𝑟𝑒𝑎𝑔 is the area of annotated region, and 𝑀 is the number of clips that anomaly occurs.
</p>