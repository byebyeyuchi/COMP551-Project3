1. files:
readh5.py: used to parse the dataset, select training sets and tests sets and store them under data folder

load_data.py: define dataloader so that we can use it to load data

train_ney.py: used to train networks. The result model will be store under output folder

generate_csv.py: load the weights models and use it to predict the test set, and eventually store the prediction result as result.csv

2. parser choice(train_net.py):
    -net: choose between resnet18, resnet34, resnet50
    -w: the number of cpu threads loading datasets
    -output: file name of weight model in output folder
    -epoch: how many rounds we train the model
    -bz: batch size

3. parser choice(generate_csv.py):
    -net: neural network you use
    -output: which weight model to load
    
Note: You need to add the MNIST_synthetic.h5 file to this folder before usage




