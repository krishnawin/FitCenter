# FitCenter

#### Research Question


Human activity classification has various applications including health monitoring, fitness tracking, and behavior analysis. Over the years, numerous tools and techniques have been developed to collect data using motion capture devices such as wearables and motion capture cameras for tracking human activity. Classification algorithms such as KNN and Support Vector Machines have been proven to accurately predict activities. However, fitness tracking presents additional challenges compared to traditional activity monitoring. Fitness exercises always involve a set of activities or poses performed in a specific manner and within a certain time frame.

There are two primary methods for tracking human activity:

Attitude Sensor: This method involves using a combination of sensors such as Gyroscope, Magnetometer, and Accelerometer to measure human activity in terms of 9 degrees of freedom (DOF).
Human Keypoint Detection using Web Cameras: This approach, particularly suitable for fitness tracking, utilizes web cameras to detect key points on the human body. It's a cost-effective and portable solution. Recent advancements in deep neural network models, such as OpenPose, Detectron, and Blazepose, leverage Convolutional Neural Networks (CNNs) to accurately predict both 2D and 3D keypoint estimation. These models offer promising capabilities for precise fitness tracking research.

### Data Sources

**Dataset:**The dataset used in this project is sourced from Kaggle and can be accessed at https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video

This dataset contains videos of people doing workouts. The name of the existing workout corresponds to the name of the folder listed.

Video format: .mp4

## Data Preparation

The dataset comprises 24 different exercises, with each exercise folder containing several video recordings of varying lengths. Due to resource constraints and data quality considerations, a smaller selection of exercises and videos has been curated for this exercise. Below is the list of workouts chosen for classification. Careful selection ensures coverage of a wide range of movements while avoiding overfitting:

Bench Press
Chest Fly Machine
Deadlift
Lateral Raise
Push-up
The curated videos can be found under the data/videos folder of this project.

## Keypoint extraction
  The Next step will be to extract the keypoints of the workouts so that we can use them in the classification algorithm. i have chosen  Detectron (from Meta) and MediaPipe(from google) for the pose estimation and keypoint extraction process.  Detectron provides only 18 keypoints in 2D and media pipe can provide 32 keypoints in 3D.
  https://detectron2.readthedocs.io/
  https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

   <img src ="https://www.kdnuggets.com/wp-content/uploads/3d-keypoints-human-pose-estimation-0.png"/>

  Below are the Jupyter Notebook for running Dectron and mediapipe against our workout video dataset

  /DataPreparation/Detectron_dataprep.ipynb
  /DataPreparation/MediaPipe_dataprep.ipynb

  the extracted datapoints are stored in the below folders
  /data/Detectron/Train
  /data/MediaPipe/Train

## **Exploratory data analysis:**

  There are no null values or duplicates found in the keypoints dataset . the full analyis can be found under the below jupyter  notebooks
  /EDA/EDA_Detectron.ipynb
  /EDA/EDA_MediaPipe.ipynb

  THe EDA confirms the correlation between different keypoint distances within  specific workouts . the line chart of different keypoints distance within a specific video of a workout correlates with other videos as well. This can be used as our basis for further modeling the prediction 

### Methodology

As discussed in our reaearch question the problem we are trying to solve is to estimate fitness tracking at a specific pose level which involves the estimation of differents join positions and also temporal. The EDA of the line chart shows a stroing correlation of the  flow between different videos within the same workout i.e the problem can be boiled down to a multivariate timeseries classification . The keypoints are the multivariables in a given timeframe and the entire video is a timeseries of different keypoints at each frame. The best algorithm for solving timeseries classification is DTW (Dynamic TIme warping) and the best library that implements it i tslearn. tslearn implements DTW and also extends Sklearn classification algorithm such as KNN and SVM to prform multivariate timeseries classification.

# what is DTW (Dynamic Time Warping)

Dynamic Time Warping (DTW) is a similarity measure between time series. Let us consider two time series x and y of respective lengths m and n. Here, all elements xi and yi are assumed to lie in the same-dimensional space.DTW between 
 x and y is formulated as the following optimization problem:
  
  <img width="351" alt="Screenshot 2024-05-09 at 1 50 00 PM" src="https://github.com/krishwin/FitCenter/assets/26986911/5c31aa98-fb89-4476-8960-7df06c4532ba">

  Instead of using the raw keypoints in dtw we can use the euclidean distance and eulear angle of various joints between frames as suggested by this paper https://link.springer.com/article/10.1007/s42979-023-02063-x  
  Euclidean distance : This is distance travelled by a joint keypoint between frames .
  Eulear angle : This is the angle or direction of the same joints between frames .

  The formula to calculate Euclidean distance is as below
    d = np.sqrt((x2-x1)^^2 + (y2-y1}^^2)
     The choosing of the number of frames to interleve becomes an hyperparameter as u can see below that it can impact the model performance and accuracy.

The above outlined technique is used in the detectron  2D dataset . please refer below notebook
/Models/Classification_Detectron.ipynb

For the Mediapipe generated 3D keypoints we can use an advanced method of calculating Quaterneon angles instead of eulear angles used above in 2D keypoints. 
This method is described in paper https://arxiv.org/pdf/2303.08657
in this method instead of using distance and angle of just the joints we can use the direction and distance of an entire part such as leg or hand . 
the steps are in the below notebook
/Models/Classification_MediaPipe.ipynb

# Results
 Below are the accuracy score of the KNN and SVM timeseries model against 2D vs 3D keypoint Dataset.
<table>
  <th>Model</th>
  <th>Detectron 2D</th>
  <th>MediaPipe 3D</th>
  <tr>
    <td>KNN</td>
    <td>.52</td>
    <td>.75</td>
  </tr>
  <tr>
    <td>SVM</td>
    <td>.32</td>
    <td>.5</td>
  </tr>
</th>
</table>

# Grid Search best parameters
<table>
  <th>Model</th>
  <th>Detectron 2D</th>
  <th>MediaPipe 3D</th>
  <tr>
    <td>KNN</td>
    <td> {'knn__metric': 'softdtw', 'knn__n_neighbors': 2, 'knn__weights': 'uniform'})</td>
    <td>{'knn__n_neighbors': 2, 'knn__weights': 'distance'}</td>
  </tr>
  <tr>
    <td>SVM</td>
    <td>{'svc__gamma': 0.1, 'svc__kernel': 'gak'})</td>
    <td>{'svc__C': 100, 'svc__gamma': 'auto', 'svc__kernel': 'gak'}</td>
  </tr>
</th>
</table>

 

