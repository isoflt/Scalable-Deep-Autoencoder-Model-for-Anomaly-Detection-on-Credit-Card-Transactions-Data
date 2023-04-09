# Scalable Deep Autoencoder Model for Anomaly Detection on Credit Card Transactions Data
Came across a dataset on Kaggle which contained 140,000 credit card transactions from Europe of which 253 are fraudulent. Naturally, this dataset is highly unbalanced, the positive class (frauds) account for 0.180% of all transactions. The dataset contains only numerical input variables which are the result of a PCA transformation, as stated in the Data Card:
> "Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features C1, C2, … C12 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise."

## Autoencoders
My choice to use a Deep Autoencoder for this dataset arose from their effectiveness in anamoly detection. A York University paper outlines this in its abstract:

> Deep autoencoders, and other deep neural networks, have demonstrated their effectiveness in discovering non-linear features across many problem domains. However, in many real-world problems, large outliers and pervasive noise are commonplace, and one may not have access to clean training data as required by standard deep denoising autoencoders. Herein, we demonstrate novel extensions to deep autoencoders which not only maintain a deep autoencoders’ ability to discover high quality, non-linear features but can also eliminate outliers and noise without access to any clean training data. Our model is inspired by Robust Principal Component Analysis, and we split the input data X into two parts, X = LD + S, where LD can be effectively reconstructed by a deep autoencoder and S contains the outliers and noise in the original data X. Since such splitting increases the robustness of standard deep autoencoders, we name our model a “Robust Deep Autoencoder (RDA)”. Further, we present generalizations of our results to grouped sparsity norms which allow one to distinguish random anomalies from other types of structured corruptions, such as a collection of features being corrupted across many instances or a collection of instances having more corruptions than their fellows. Such “Group Robust Deep Autoencoders (GRDA)” give rise to novel anomaly detection approaches whose superior performance we demonstrate on a selection of benchmark problems.

More here: https://www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p665.pdf

Since we are trying to distinguish the anamolous transactions from the ones that are not, a deep autoencoder is a robust option.

## Business Problem
Effectively this is a major business problem for large financial institutions. WalletHub cites that in 2020 global credit card and debit card fraud totaled $28.85 billion and card issuers (banks) incurred 88% of those losses, representing a staggering amount of losses at a global scale. As such the business case would be as follows:
- To assess if a transaction is risky (in other words, fraudulent) from the given credit card data.
- Learn from various features of normal transactions to distinguish fraud transactions better.

The machine learning case would be to develop a machine learning model based on deep autoencoders to learn distribution and relation between features of normal transactions. And finally, we need to deploy the model in a scalable way so that business decisions can be taken in near real-time in assessing riskiness of a transacttion.

More fraud statistics here: https://wallethub.com/edu/cc/credit-debit-card-fraud-statistics/25725

## Scalability
The goal was to build a scalable solution, and building an API endpoint in Python for the model would require using a Flask deployment, but Flask cannot handle a very high RPS (requests per second), hence I used Gunicorn. Gunicorn is a WSGI server which scales existing Flask deployments so it can handle an RPS of more than a 100, or even 1000. Vanilla Flask cannot handle such a high RPS, hence building a scalable solution, albeit basic, was important since credit card transaction volume exceeds 1 billion transactions per day.

## Running the Program
The source directory (src) contains the `Engine.py` file which is the main entry point into the code. `ML_Pipeline` contains all the auxiliary files, each one run accordingly to what is selected in the Engine. Roughly, there are 3 components to the project: `input`, `output`, and `src`. The `input` folder contains the data, the `output` folder contains the model and other model outputs to be used for deployment, and `src` contains all the source/code files for the program, including the machine learning pipeline. The `lib` folder contains a notebook (`Deep-Autoencoder.ipynb`) exploring the dataset, and more importantly, the `Model_Api.ipynb` notebook for accessing the API endpoint from the client side.
