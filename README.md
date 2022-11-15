# A Contrastive Pretrain Model with Prompt Tuning for Multi-center Medication Recommendation

This is the implementation of the paper "A Contrastive Pretrain Model with Prompt Tuning for Multi-center Medication Recommendation".

You can implement our model according to the following steps:

1. Apply for and download the [eICU](https://eicu-crd.mit.edu/about/eicu/) dataset. Then, please unzip the data and put the `diagnosis.csv`, `medication.csv`, `patient.csv` and `treatment.csv` into the path `data/eicu/raw/`.
2. Run the pre-processing scripts to clean up the dataset.

   ``

   python data/processing-eicu.py

   ``
   Then, run the following scripts to filter out the hosiptal with too few records:

   ``python data/filter_hospitals.py ``

   After the pre-processing, you can get the pre-processed data in `data/eicu/handled`. We repeatedly split the data 5 times and using the random seed [42, 43, 44, 45, 46]. If you want to change the random seeds, please refer to `data/processing-eicu.py`.
3. Install the necessary packages. Run the command:

   ``

   pip install -r requirements.txt

   ``
4. To get the pretrain models of the five split data, please run the command:

   ``

   bash ./experiments/pretrain.bash

   ``
5. Finally, you can run the following bash to train and test proposed TEMPT:

   ``

   bash ./experiments/prompt.bash

   ``

Note: the results of all hospitals and mutiple runs will be saved in `./log/results/`. You can use the `analysis.ipynb` to analyze the results and get the average performance.
