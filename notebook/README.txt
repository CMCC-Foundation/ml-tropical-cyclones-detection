Notebook 1 – inference_noteook.ipynb

This notebook is designed to perform inference using the trained model for TCs detection and to apply a tracking algorithm to identify the trajectories of the detected systems.  

Workflow  

1. Parameter definition
    For first, the user specifies:  
        - `main_dir`: root directory of the project.  
        - `dataset_dir`: path to the climate dataset to be analyzed (e.g., CMIP6, NICAM, ERA5).  
        - `model_dir`: path to the pre-trained model to be used for inference.  
        - `ibtracs_src`: path to the **IBTrACS** file used as ground truth for validation.  
        - `year`: the year on which inference will be performed.  
        - `device`: compute device (`cpu`, `cuda`, `mps`, etc.).  
        - `lat_range` / `lon_range`: geographical coordinates of the study domain. 

2. Model and dataset loading
    Standard cells are provided to:  
        - Load the pre-trained model  
        - Load the input dataset  
        - Prepare the data for inference  

3. Inference
    The model is executed on the selected region and year to detect TC occurrences.  

4. Tracking
    The detections are post-processed with a tracking algorithm in order to reconstruct the actual TC tracks across space and time.  

5. Visualization
    The notebook provides several visualization functions for:
        - Visualize detections against observations.
        - Display predicted and observed tracks.
        - Show performance metrics (POD, FAR).
        - Plot track duration distributions.

Output  

- Tropical cyclone detections from the model  
- Tracked storm paths  
- Performance merìtics (POD, FAR; track durations)

Notebook 2 - results_analysis.ipynb

This notebook is designed to analyze the performance of the trained model for TCs detection and tracking.  
It compares the model outputs against the IBTrACS reference dataset, computing localization errors, classification metrics, and track matching statistics.  

Workflow  

1. Setup
    - Import required libraries and custom modules  
    - Define the inference directory (where detection results are stored)  
    - Select the model to analyze  
    - Load the IBTrACS dataset and define test years  

2. Data loading
    - Load model inference results (CSV files) for the selected years  
    - Merge and preprocess detection data  
    - Load and preprocess IBTrACS observations (coordinates rounded to training grid)  

3. Localization analysis
    - Match detections and observations by time  
    - Compute spatial distances (Haversine distance)  
    - Filter detections within a maximum allowed distance  
    - Report statistics: min, max, mean, and median localization errors  

4. Classification analysis
    - Compute True Positives (TP), False Positives (FP), False Negatives (FN)  
    - Calculate precision, recall, F2-score  
    - Print detailed classification results  

5. Tracking
    - Apply the tracking algorithm to detections  
    - Compare the number of detected tracks vs observed tracks  
    - (Optional) visualize tracks with plotting functions  

6. Track matching
    - Match detected vs observed tracks  
    - Compute metrics: Hits (H), Misses (M), False Alarms (FA), Probability of Detection (POD), False Alarm Rate (FAR)  

7. Results saving
    - Store performance metrics into a CSV file inside the model’s inference directory  
    - Include localization, classification, and tracking statistics  

Output  

- Localization error statistics (km)  
- Classification metrics (precision, recall, F2-score, TP/FP/FN counts)  
- Tracking results (number of detected vs observed tracks)  
- Track matching metrics (POD, FAR, Hits, Misses, False Alarms)  
- Results summary saved as `results_analysis.csv`