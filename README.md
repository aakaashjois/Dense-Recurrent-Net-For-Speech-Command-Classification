# Dense Recurrent Net For Speech Command Classification

This was a course project for [Audio Content Analysis](https://wp.nyu.edu/jpbello/teaching/aca/) taught by Prof. Bello. It is based on a [Kaggle competition](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge) to classify one-second long English speech commands.

Three models were implemented in Keras:
- a ConvNet
- a DenseNet
- a Dense-Recurrent Net

The Kaggle dataset was transformed with simple Gaussian noise to make a noisy variant of the clean dataset. The following table lists the multi-class accuracies of the three models on both clean and noisy datasets:

|  Architecture |     Architecture    | ConvNet | DenseNet | Dense-Recurrent Net |
|:-------------:|:-------------------:|:-------:|:--------:|:------------------:|
| Clean dataset |  Training accuracy  |  85.68% |  90.28%  |       99.71%       |
|               | Validation accuracy |  80.47% |  84.74%  |       83.45%       |
|               |   Testing accuracy  |  81.49% |  85.04%  |       83.39%       |
| Noisy dataset |  Training accuracy  |  88.02% |  88.67%  |       99.59%       |
|               | Validation accuracy |  83.64% |  81.95%  |       82.74%       |
|               |   Testing accuracy  |  84.14% |  83.20%  |       82.60%       |

A copy of the [report](./Report/Report.pdf) is available for reference.
