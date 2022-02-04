# DeCA
This is an implementation for our WWW 2022 paper **Learning Robust Recommenders through Cross-Model Agreement**.  

## Requirements
+ torch == 1.9.0+cu102
+ Numpy 
+ python3

## Datasets

+ Movielens-100k: We provide the [link](https://grouplens.org/datasets/movielens/100k/) to the original data and also the processed dataset in the folder `data`. 

+ Modcloth and Electronics: These two datasets were first processed by the paper [``Addressing Marketing Bias in Product Recommendations``](https://dl.acm.org/doi/10.1145/3336191.3371855). Then we converted them into our format. If you need to use this dataset, you may also need to cite this paper. 

+ Adressa: This dataset is from [``DenoisingRec``](https://github.com/WenjieWWJ/DenoisingRec), and it's already our format. If you use this dataset, you may also need to cite the paper [``Denoising Implicit Feedback for Recommendation.``](https://arxiv.org/abs/2006.04153). 


## Parameters
Key parameters are all provided in the file ``configs.py``, and you can let the code choose the specific parameters for the model and the dataset with "python xxx.py --default". 


## Commands
We provide following commands for our methods `DeCA` and `DeCA(p)`. 
Simply run the code below will return the results shown in the paper:
```
python main.py --model GMF --dataset ml-100k --method DeCA --default
```
where `--default` means using the default setting. `--model` is the model drawn from `GMF, NeuMF, CDAE, LightGCN`, `--dataset` should be in `ml-100k, modcloth, adressa, electronics`, `--method` need to be in `DeCA, DeCAp`. Remove the `--method` term, the code will run normal training.  
If you want to use your own settings, try:
```
python main.py --model GMF --dataset modcloth --C_1 1000 --C_2 10 --alpha 0.5 --method DeCA
```


## Citation
If you use our codes in your research, please cite our paper.

