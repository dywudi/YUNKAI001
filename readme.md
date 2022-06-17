# Action Recognition
## Prepare environment
It is recommended to run CentOS with a GPU with 12GB or more of video memory, other platforms like Windows is not recommended but should be fine, not tested.

```pip install requirements.txt```

## Demo

python inference_video.py

## train from scratch
Since the dataset is very big, we put the dataset on cloud, the dataset is still uploading when this readme is written. Please consult us for the download links.
After that, Run these three commands in turn from the command line to retrain visual model. During the process, you may establish some new folders.
Create traintestsplit.
```python create_train_test.py``''
To create the label (which can take a long time on a non-GPU driven computer).
```python create_label.py``
Run the training (which can be long on a non-GPU driven computer): ```python train.py
```python train.py``
Testing
```python evaluation.py``

For LSTM model, please run ```python lstm_train.py``. 
