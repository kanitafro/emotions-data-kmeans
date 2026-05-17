# Datasets

## 1. Bigger dataset with 6 labels

The dataset used for training on 6 labels has more than 450k instances, but is consisted of a few datasets (5 from Kaggle and 1 custom dataset `more_surprise.csv` that is located in this folder).

### Main dataset (all 6 labels)
Taken from Kaggle - [click here for source](httpswww.kaggle.comdatasetskushagra3204sentiment-and-emotion-analysis-dataset)

 Labels joy, sadness, anger, fear, surprise, love
 Size 422k samples
 Columns `text` and `label`

INSTRUCTIONS
1. Download the dataset, name the file `dataset_6labels.csv` and put it here in dataraw.
2. After that, you will download the other 4 datasets from Kaggle, listed below, put them into this folder and run the notebook `merge_love_surprise.ipynb`.
    After running the notebook, the result is the file `love_surprise_bonus.csv` in dataraw.
3. Run the script clean_dataset.py by calling 
    ```
    python -m scripts.clean_dataset
    ```
    This script will produce files dataprocesseddataset_6labels_clean.csv (422k) and dataprocesseddataset_6labels_clean_more.csv (450k+)
4. You're ready to train on 6 emotions.
---

### Extra dataset 1 [Kaggle](httpswww.kaggle.comdatasetspraveengoviemotions-dataset-for-nlp) (only love and surprise)
file path datarawextra_dataset_Praveen.csv

 Instances extracted  
  - `love` - 1641  
  - `surprise` - 719

NOTE Download the dataset, name the file `extra_dataset_Praveen.csv` and put it here in dataraw. The file path indicated above is the path called in `merge_love_surprise.ipynb`. 

---

### Extra dataset 2 [Kaggle](httpswww.kaggle.comdatasetspashupatiguptaemotion-detection-from-text) (only love and surprise)
file path datarawextra_dataset_PashupatiGupta.csv

 Instances extracted  
  - `love` - 3842 
  - `surprise` - 2946

NOTE Download the dataset, name the file `extra_dataset_PashupatiGupta.csv` and put it here in dataraw. The file path indicated above is the path called in `merge_love_surprise.ipynb`.

---

### Extra dataset 3 [Kaggle](httpswww.kaggle.comdatasetssimaanjaliemotion-analysis-based-on-textdata) (only love and surprise)
file path datarawextra_dataset_SimaAnjali.csv

 Instances extracted  
  - `love` - 39553
  - `surprise` - 6954

NOTE Download the dataset, name the file `extra_dataset_SimaAnjali.csv` and put it here in dataraw. The file path indicated above is the path called in `merge_love_surprise.ipynb`.

---

### Extra dataset 4 [Kaggle](httpswww.kaggle.comdatasetsbryanhuertasocial-media-sentiment-data) (only love and surprise)
file path datarawextra_dataset_BryanHuerta.csv

 Instances extracted  
  - `surprise` - 31  
  - `love` - 26

NOTE Download the dataset, name the file `extra_dataset_BryanHuerta.csv` and put it here in dataraw. The file path indicated above is the path called in `merge_love_surprise.ipynb`.

---

### Custom dataset (only surprise)

I generated 663 instances for surprise by prompting LLMs (ChatGPT and Deepseek) giving them special sublabels as guidance (e.g. confusion, excitement) and varying them in length.

All 663 new instances that only have label `surprise` are located in this folder, in the file `more_surprise.csv`.

---

## 2. GoEmotions dataset (27 emotions + neutral)

Instructions to get the data and encode the labels
