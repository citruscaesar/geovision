# The Flow of Data

## Extraction, Transformation and Storage: geovision.data 
Datasets are downloaded either from their hosted sources from the web or from the personal cloud hosted storage server, which is Contabo currently. This idea commonly called data warehousing, where
it is cleaned, homgenized and stored in a common data format ready for analysis. The advantages of using the latter is that:

1. the dataset is already prepared before storage, so the transformation step can be entirely skipped if the task-specific dataset is stored or involve relatively few steps if a more general 
   form of the dataset is stored to avoid redundant copies of the same data.
2. the etl pipeline from the orginal data source might be slow with inconsistent network speeds, slow transformation steps, etc.
3. although uncommon, sometimes the web hosted sources today might not be online in the future, thus it's prudent to store a copy of the data as a backup.

The extracted/raw data is placed in a temporary subdirectory for cleaning. At this point, an exploratory analysis is performed to look for any patterns, inconsistencies and redundancies in the 
organization, size, formating, metadata, etc. to help envision the transformation steps required to convert the data into a consistent schema. Further, visual inspection of the  data, i.e. staring at
it, helps not only to write the automated scripts in later steps, but also to familiarize with the data itself. 

The data cleaning steps are organized in a sequential and modular way, such that at the end of each step, an metadata.h5 contains all the metadata about the dataset, and is used 
by the later steps to load and transform either just the metadata itself, or both the dataset and the metadata if needed. Some datasets often come with large amounts of associated metadata, which
are stored in tables in a relational manner, with a integer index acting as the primary/foreign key assigned to each unique data sample. Even when the datasets don't have these metadata, they are 
either downloaded using other sources, or computed as part of EDA or in later descriptive analysis steps, and are stored alongside the indexed images to be joined as needed. 
* metadata.h5
  * index (table): [:image_{path/name}, :(label_idx(s)/label_bbox/mask_path), ...] 
  * spatial: (table): [:image_width, :image_height, :crs, :x_offset, :y_offset, :x_gsd, :y_gsd, ...]
  * spectral: (table): [:channel_statistics (min, max, mean, variance, skewness, kurtosis), :histogram_based_statistics (entropy, energy, homogeniety), ...]
  * temporal: (table): [:time_of_capture, :time_of_overpass, ...]
  * additional: (table): [:radiometric_information, :sensor_calibration_information, :data_format_and_decoding_information, ...]

The cleaning steps should result in a task specific dataset with a clearly defined task, such as multiclass classification, multilabel classification, binary segmentation, etc., which differs from
more generalized versions of the dataset, such as a localization dataset with multiple bboxes per image which can always be used as a multilabel classification dataset, stored with both kinds of
labels simultaneously. It should also be stored in one of several storage formats (e.g. Imagefolder, Memory Mapped Numpy Arrays, HDF5, Zarr, LMDB, etc.) with the metadata attatched as tables.
It is clear then that the extraction and transformation steps are (mostly) unique to each dataset and it's intended purpose, thus it's not wise to try to generalize them.

Image datasets are often stored as flat image files on the filesystem, and distributed by zipping them into single or multivolume archives. Here this format is called the Imagefolder format, which
can accomodate images of different sizes (H, W, C), different encodings (e.g. jpg, png, tiff) as long as they can be decoded by the dataloading program and are easy to view using any image viewer 
application. However, while compressed encodings like JPEG save valuable space on disk, they introduce some overhead as each image has to be decoded every time they are loaded, this is a problem
for larger datasets especially, as smaller datasets in this format can be loaded once and held in (page-locked) memory for the duration of the training-evaluation process. The big problem with
this style is that they don't work well with cloud storage (millions of small files being requested over the network), and have to be compressed into chunks and decompressed every time they have to be 
transferred, which adds additional CPU overhead. 

Another way to image datasets are stored are using data formats such as HDF5, which are optimized to store and load slices from multidimensional arrays on disk, and are significantly faster than the 
Imagefolder style. A distinct advantage such formats offer is that they are easier to stream / download from cloud storage providers as they can be broken up into chunks to optimize read speeds, 
and work really well with OS-level caching for a further boost.The obvious disadvantage is that the images are constrained to be in an array, and thus have to be resampled to the same H, W and C.
Another disadvantage is that these formats take up way more space as the data is stored as raw f32 NxCxHxW arrays. A way to mitigate these is that the images can be encoded as jpg and the bytestream
is stored as an array of length N, but may result in quality loss and re-introduces the decoding overhead. A desirable requirement that such file formats help with is that for satellite data, we 
often also want to slice the data along the H, W and C dimensions too, which is requires us to load the entire image and then crop, incase of imagefolder / mmapped datasets.
* dataset.h5
  * images (array): 4d array of images
  * masks (array): 1d array of encoded masks, if required 
  * class_names (array): 1d array of string class_names in sorted order (label_idx depends on the order)
  * index, spatial, spectral, temporal, ... (table):
  * {additional_metadata} (alphanumeric): additional information such as common :crs, :gsd, etc.

In any case, the data is stored under a subdirectory inside the dataset root, with the metadata inside the dataset file, or separate metadata.h5 in case of the Imagefolder style, containing all the 
information for the cleaned and indexed dataset in that subdir with image paths, label_information in /index and other data in /spatial, /temporal etc. The same directory structure is mirrored in 
cloud storage, with the bucket name as the dataset name.

The Dataset class is responsible for abstracting away all this complexity by implementing the __getitem__ method, which returns a data sample based on it's integer index. It implements subsampling and
augmentation over the dataset, as defined in the dataset configuration before training, which involves splitting the dataset into train-val-test splits (tabular sampling based on strategy chosen),
sampling specific H,W crops from the dataset  (spatial sampling based on the image or a region on earth), specific channels (spectral sampling specific bands in case of multispectral/hyperspectral
images), specific time periods (temporal sampling, e.g. to capture seasonal variations) and then applying augmentations to increase robustness / mitigate overfitting (bias variance tradeoff). Another 
feature of the dataset class is that it can combine with other datasets of the same task and number of classes (+) to yeild samples from either dataset, e.g. building footprint segmentation datasets
with 2 classes, or with other datasets (*) to yeild samples over the same region simultaneously, like Sentinel-2 and Worldview-2,3 for super-resolution.

NOTE: Looking into other formats such as Zarr (often touted as the modern HDF5) or LMDB should be worthwhile.
NOTE: write about STAC
NOTE: write about dataloading and maximizing throughput
NOTE: write about data augmentation, applying only geometric augmentations to localization based models, CutMix and MixUp

## Data Analysis and Model Understanding: geovision.analysis
The analysis of a dataset before training aims to highlight any significant patterns present, which then helps choose various hyperparameters and preprocessing procedures during training.
The insights from this analysis directly helps the researcher get an idea about the data distribution and inform the choice of model architecture, objective fn, metrics, dataset sampling and 
augmentation procedures. The analysis of a trained model aims to highlight the strengths and weaknesses of the model, it's biases and lend some insight to why a it might be underperforming and 
where it needs to be improved. 

### Data Analysis Methods and Expected Insights
Often little to do with label information, more about the images themselves, should be able to work with any index_df(raw/cleaned/subsampled).
#### Dimension Analysis
### ...
Dimension information such as image dimensions -> used for cropping, padding and aspect ratio problems -> also to calculate file size in memory
File size on disk information for flat (compressed) files (.jpg, .png, .tiff)
Color information like channel wise min, max, mean, median, variance, skewness and kurtosis -> sampling distribution of sample statistics -> color normalization, transformation, augmentation strategies
  ... channel correlation, pearson's, spearman's, mutual information, partial correlation,
  ... histogram based statistics, entropy, energy, homogeniety
  ... other methods like color space transformation, spatial color analysis, principal components, etc.
Spatial information like crs and gsd, autocorrelation, semivariogram, endmember extraction, anomaly detection, etc.

#### Model Evaluation and Understanding 
Model Performance Evaluation
script to pass a dataset through the model and store its outputs in a file
  -> can be used to find for which examples does the model perform well/poor and analyze why 
  -> can be used to visualize embeddings (after dimensionality reduction, like t-SNE or UMAP)
{model}_{dataset}_evaluation.h5
  training_hparams: (dict or yaml or something)
  model_architecture: (string)
  {embeddings}: (ordered array(s) of vector embeddings)
  {attribution_masks}: (ordered array(s) of (compressed if possible) masks)
  metrics: (table)
      :idx (foreign key -> metadata index)
      :{metrics per image}

## Model Training and Fine-Tuning
Experiment Configuration -> brief overview
Logging and Tracking Experiments
Encoder Decoder Architecture
Custom Objective Functions / Metrics

## Architecture Optimization
Pruining
Quantization

## Deployment
Export to ONNX, torch.compile



