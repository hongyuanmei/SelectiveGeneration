# SelectiveGeneration
Code base for [Mei et al. NAACL 2016 paper](https://arxiv.org/abs/1509.00838/)

## Dependencies
* [Anaconda](https://www.continuum.io/) - Anaconda includes all the Python-related dependencies
* [Theano](http://deeplearning.net/software/theano/) - Computational graphs are built on Theano
* [NLTK](http://www.nltk.org/) - Natural Language Toolkit

## Instructions
Here are the instructions to use the code base

### Prepare Data
Creat a folder called 'data' and copy data files (to be added) to it

### Train Models
To train the model with options, use the command line 
```
python train_models.py --options
```
For the details of options, please check
```
python train_models.py --help
```

### Test Models
Choose a model to evaluate on dev or test set, with the command line:
```
python test_models.py --options
```
For the details of options, please check
```
python test_models.py --help
```

## License

This project is licensed under the MIT License - see the [License.md](License.md) file for details

