Dataset Overview
·  IMDB Dataset: Directly obtained from the built-in tensorflow.keras.datasets.imdb library. After merging all available data, the dataset was split into training, validation, and test sets in a ratio of approximately 6:2:2.
·  Amazon, SST-2, and Ceramic Datasets: These datasets are provided as CSV files.


Feature Extraction Strategy
·  Considering CPU memory limitations, processing speed, and the need to perform multiple experiments across different datasets, we first extracted text    embeddings using pre-trained models such as RoBERTa and XLNet.
·  During the training phase, these precomputed embeddings are directly used as model inputs, significantly improving training efficiency.


Model Download and Configuration
In the feature extraction phase, you need to:
·  Select the dataset you wish to train on.
·  Download the corresponding pre-trained models (such as RoBERTa, XLNet, etc.) from Hugging Face.
·  Configure the max_length parameter according to your needs.
·  Specify the corresponding file paths to save the extracted local and global embeddings.


Instructions for Baseline and Proposed Model Training
When using the baseline model scripts and the proposed model ("Ours") script:
·  Ensure that the dataset paths, feature embedding storage paths, and max_length settings are correctly matched to the dataset in use.
·  Select the appropriate activation function and loss function based on whether the task is binary classification or multi-class classification.
·  In the "Ours" model, after extracting the local and global features, manually reshape the global feature tensors.
   The reshape dimensions must match the number of samples in the training, validation, and test sets for each dataset.


Instructions for Ablation Studies
For ablation experiments:
·  Modify the "Ours" model by either removing specific feature modules according to the descriptions in the paper, or
·  Feed the extracted feature vectors into different feature extraction networks as required.
·  Adjustments should be made based on specific ablation settings, such as disabling certain components (e.g., CNN, BiLSTM, or attention mechanisms) to evaluate    their impact on overall performance.


Contact
If you encounter any questions during the experiments, please feel free to contact me via email at 18068016991@163.com.