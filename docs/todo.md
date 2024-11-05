### geovision/
- [] use pixi instead of conda+poetry, remove poetry dependencies and recreate environment

### geovision.experiment/
- [x] add lr_warmup to scheduler
- [] add stocastic weight averaging
- [x] add train_lr_epoch to experiment_logger, figure out where to read lr from litmodule.schedulers()[-1].get_last_lr()[0]
- [] cmd tool to live plot metrics from csv / feather / hdf
    -> test how metrics.csv looks when the training process has hiccups (like crashing, resuming from a ckpt, etc.)
    -> if it looks good, write a script to sync the metrics.csv at a fixed time interval (e.g. 5sec) and update the graph
    -> if not, figure out how to get a metrics.csv from experiment.h5 while it's being written to, or another .csv/.paraquet based ExperimentWriter class
- [x] script to plot all metrics within a run (optionally to specified directory)
    -> take project_name, run and (...) as args
    -> option to list runs based on project names
    -> default view should be the confusion matrix, loss and configured metric plots
    -> option to plot a metric and compare it across runs

### geovision.io/
- [] SSHIO to sync the code, .env file and experiment logs b/w the local and the remote machine
    -> scripts to upload all the code including the .env file and create the directory structure ready
    -> fn to periodically download log files
    -> Fabric or Paramiko
- [] S3IO (calls s5cmd) to download datasets and download/upload ckpts and weights

### geovision.models/ 
- [] write loaded config to hparams.yaml at the beginning of each run (pl_module.save_hyperparameters())
- [] list and add basic building blocks (e.g. vgg, residual, dense, mbconv)
    -> as generalized as possible
    -> add optional weight init, ideally external to the class itself
    -> expect or not expect to downsample/upsample the images
- [] add commonly used architectures as generally as possible
    -> each layer should have a unique name and be accessible from the outside (important for attribution)
- [] add constructors (fn not methods) to define a model using specific hyperparameters and load weights
- [] add common weight init methodology (from torchvision.models/fastai/timm/huggingface/self-made etc.)
    -> port weights once and save for later use
- [] use LightningModules to define common workflows, like Classification, GAN, VAE, MoCo etc. with training, evaluation and inference
- [] add Potts and Normalized Cut Loss (for polygon regularization)

### geovision.analysis/
- [] attribution Methods like Feature Visualization, Guided Backprop, IG, GRAD-CAM, LIME, SHAP, etc.
    -> (optionally) implement from scratch without using Captum 
- [] script to compute and store image statistics right in the metadata dataframe files, aggregate to compute dataset statistics
    -> load each image, calculate channel wise mean and variance and store in a dataframe -> concat with metadata to calculate aggregate stats dynamically for any split as properties
- [] classical statistics based EDA for datasets to highlight geometry and spectral characteristics

### geovision.data/
- [] add Inria
- [] add Pascal VOC, MS COCO, OxfordIIITPets
- [] massachussets, ISPRS, SpaceNet, CrowdAI 
- [] add spatial sampler, at the image level (all images) and spatial level (georegistered scenes) [ref. Geo-Tiling for Semantic Segmentation]
    -> useful for inference time when entire images cannot be fed into the model
- [] add FMoW, BigEarthNet and HySpec11k (for pretraining)
- [] add PASTIS-R for agricultural analysis
- [] figure out why CutMix nd MixUp aren't working
- [] download Worldview Imagery from ESA Everyday (Quota of 4 images / day)
- [] download PRISMA Imagery Everyday
- [] download EnMap Imagery Everyday
