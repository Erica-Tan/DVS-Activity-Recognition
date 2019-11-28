# Dynamic vision sensor for action recognition

## Requirements

* Python 3.7
* virtualenv
* cuda 10

## Dependencies
Create a virtual environment with `python3.7` and activate it

    virtualenv venv -p /usr/local/bin/python3.7
    source venv/bin/activate

Install all dependencies by calling 

    pip install -r requirements.txt


## Training
Before training, download the DVS Action Recognition Dataset from Github repository https://github.com/MSZTY/PAFBenchmark
    
Then start training by calling

    python main.py --validation_dataset ../dataset/ActionRecognitionAVI2Npy/validation --training_dataset ../dataset/ActionRecognitionAVI2Npy/training --log_dir logs

### Additional parameters 
* `--device` controls on which device you want to train
* `--pin_memory` wether to pin memory or not
* `--num_worker` how many threads to use to load data
* `--num_epochs` number of epochs to train
* `--save_every_n_epochs` save a checkpoint every n epochs.
* `--batch_size` batch size for training


## Testing
Once trained, the models can be tested by calling the following script:

    python testing.py --test ../dataset/ActionRecognitionAVI2Npy/testing --checkpoint ./checkpoint/checkpoint_00000_2.2948.pth --device cuda:0
