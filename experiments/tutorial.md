# Tutorial
##### for Including a Dataset into the Framework

## Introduction
This tutorial aims at providing a muster routine for including a new dataset into the framework in order to 
use the included models and algorithms with it.\
The tutorial and toy dataset (under `toy_exp`) are in 2D, yet the switch to 3D is simply made by providing 3D data and proceeding 
analogically, as can be seen from the provided LIDC scripts (under `lidc_exp`).

Datasets in the framework are set up under `medicaldetectiontoolkit/experiments/<DATASET_NAME>` and
require three fundamental scripts:
1. A **preprocessing** script that performs one-time routines on your raw data bringing it into a suitable, easily usable 
format.
2. A **data-loading** script (required name `data_loader.py`) that efficiently assembles the preprocessed data into
network-processable batches.
3. A **configs** file (`configs.py`) which specifies all settings, from data loading to network architecture. 
This file is automatically complemented by `default_settings.py` which holds default and dataset-independent settings.

## Preprocessing
This script (`generate_toys.py` in case of the provided toy dataset, `preprocessing.py` in case of LIDC) is required
to bring your raw data into an easily usable format. We recommend, you put all one-time processes (like normalization, 
resampling, cropping, type conversions) into this script in order to avoid the need for repetitive actions during 
data loading.\
For the framework usage, we follow a simple workload separation scheme, where network computations
are performed on the GPU while data loading and augmentations are performed on the CPU. Hence, the framework requires 
numpy arrays (`.npy`) as input to the networks, therefore your preprocessed data (images and segmentations) should 
already be in that format. In terms of data dimensions, we follow the scheme: (y, x (,z)), meaning coronal, sagittal, 
and axial dimensions, respectively.

Class labels for the Regions of Interest (RoIs) need to be provided as lists per data sample.
If you have segmenation data, you may use the [batchgenerators](https://github.com/MIC-DKFZ/batchgenerators) transform 
ConvertSegToBoundingBoxCoordinates to generate bounding boxes from your segmentations. In that case, the order of the 
class labels in the list needs to correspond to the RoI labels in the segmentation.\
Example: An image (2D or 3D) has two RoIs, one of class 1, the
other of class 0. In your segmentation, every pixel is 0 (bg), except for the area marking class 1, which has value 1, 
and the area of class 0, which has value 2. Your list of class labels for this sample should be `[1, 0]`. I.e.,
the index of the RoI's class label in the sample's label list corresponds to its marking in the segmentation shifted 
by -1.\
If you do not have segmentations (only models Faster R-CNN and RetinaNet can be used), you can directly provide bounding
boxes. In that case, RoIs are simply identified by their indices in the lists: class label list `[cl_of_roi_a, cl_of_roi_b]` 
corresponds to bbox list `[coords_of_roi_a, coords_of_roi_b]`.

Please store all your light-weight information (patient id, class targets, (relative) paths or identifiers for data and seg) about the
preprocessed data set in a pandas dataframe, say `info_df.pkl`. 

## Data Loading
The goal of `data_loader.py` is to sample or iterate, load into CPU RAM, assemble, and eventually augment the preprocessed data.\
The framework requires the data loader to provide at least a function `get_train_generators`, which yields a dict
holding a train-data loader under key `"train"` and validation loader under `"val_sampling"` or `"val_patient"`, 
analogically for `get_test_generator` with `"test"`.\
We recommend you closely follow our structure as in the provided datasets, which includes a data loader suitable for 
sampling single patches or parts of the whole patient data with focus on class equilibrium (BatchGenerator,
used in training and optionally validation) and a PatientIterator which is intended for test and optionally valdiation and
 iterates through all patients one by one, not discarding 
any parts of the patient image. In detail, the structure is as follows.

Data loading is performed with the help of the batchgenerators package. Starting from farthest to closest to the 
preprocessed data, the data loader contains:
1. Method `get_train_generators` which is called by the execution script and in the end provides train and val data loaders.
 Same goes for `get_test_generator` for the test loader.
2. Method `load_dataset` which reads the `info_df.pkl` and provides a dictionary holding, per patient id, paths
 to images and segmentations, and light-weight info like class targets.
3. Method `create_data_gen_pipeline` which initiates the train data loader (instance of class BatchGenerator),
assembles the chosen data-augmentation procedures and passes the BatchGenerator into a MultiThreadedAugmenter (MTA). The MTA
is a wrapper that manages multi-threaded loading (and augmentation).
4. Class BatchGenerator. This data loader is used for sampling, e.g., according to the scheme described in 
`utils/dataloader_utils.get_class_balanced_patients`. It needs to implement a `__next__` method providing the batch; 
the batch is a dictionary with (at least) keys: `"data"`, `"pid"`, `"class_target"` (as well as `"seg"` if using segmentations).
    - `"data"` needs to hold your image (2D or 3D) as a numpy array with dimensions: (b, c, y, x(, z)), where b is the 
    batch dimension (b = batch size), c the channel dimension (if you have multi-modal data c > 1), y, x, z are 
    the spatial dimensions; z is omitted in case of 2D data.
    - `"seg"` has the same format as `"data"`, except that its channel dimension has always size c = 1.
    - `"pid"` is a list of patient or sample identifiers, one per sample, i.e., shape (b,).
    - `"class_target"` which holds, as mentioned in preprocessing, class labels for the RoIs. It's a list of length b, holding
    itself lists of varying lengths n_rois(sample). 
    **Note**: the above description only applies if you use ConvertSegToBoundingBoxCoordinates. Class targets after batch 
    generation need to make room for a background class (network heads need to be able to predict class 0 = bg). Since, 
    in preprocessing, we started classes at id 0, we now need to shift them by +1. This is done automatically inside
    ConvertSegToBoundingBoxCoordinates. That transform also renames `"class_target"` to `"roi_labels"`, which is the label
    required by the rest of the framework. In case you do not use that transform, please shift and rename the labels
    in your BatchGenerator.
5. Class PatientIterator. This data loader is intended for testing and validation. It needs to provide the same output as 
above BatchGenerator, however, initial batch size is always limited to one (one patient). Output batch size may vary 
 if patching is applied. Please refer to the LIDC PatientIterator 
to see how to include patching. Note that this Iterator is not supposed to go through the MTA, transforms (mainly 
ConvertSegToBoundingBoxCoordinates) therefore need to be applied within this class directly.


## Configs
The current work flow is intended for running multiple experiments with the same dataset but different configs. This is
done by setting the desired values in `configs.py` in the data set's source directory, then creating an experiment
via the execution script (`exec.py`, modes "create_exp" or "train" or "train_test"), which copies a snapshot of configs, 
data loader, default configs, and selected model to the provided experiment directory.

`configs.py` introduces class `configs`, which, when instantiated, inherits settings in `default_configs.py` and adds 
model-specific settings to itself. Aside from setting all the right input/output paths, you can tune almost anything, from
network architecture to data-loading settings to train and test routine settings.\
Furthermore, throughout the whole framework, you have the option to include server-environment specific settings by passing
argument `--server_env` to the exec script. E.g., in the configs, we use this flag, to overwrite local paths by the
paths we use on our GPU cluster.  