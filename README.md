# Dynamic vision sensor for action recognition

## Requirements

* Python 3.7
* virtualenv

## Dependencies
Create a virtual environment with `python3.7` and activate it

    virtualenv venv -p /usr/local/bin/python3.7
    source venv/bin/activate

Install all dependencies by calling 

    pip install -r requirements.txt


## Training
Before training, download the DVS Action Recognition Dataset from Github repository https://github.com/MSZTY/PAFBenchmark
    
Convert the dataset from aedat to npy by calling

    python aedat_to_avi.py
    python avi_to_npy.py

Then start training by calling

    python main.py --validation_dataset ./dataset/ActionRecognitionAVINpy/validation --training_dataset ./dataset/ActionRecognitionAVINpy/training --log_dir ./logs

### Additional parameters 
* `--device` controls on which device you want to train
* `--pin_memory` wether to pin memory or not
* `--num_worker` how many threads to use to load data
* `--num_epochs` number of epochs to train
* `--save_every_n_epochs` save a checkpoint every n epochs.
* `--batch_size` batch size for training

