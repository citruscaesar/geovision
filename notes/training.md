# Draft
1. LightningModules for CNNs, GANs, VAEs, Autoencoders, etc.
	1. Pytest? Maybe unnecessary since they are implemented once and almost never changed
	1. add `test_classification_workflow(datamodule, litmodule, config, metric_collection, limit_batches)` which runs one epoch with the dataset, on a simple training and inference loop and computes metrics to serve as a benchmark and sanity checker.
	1. add `test_segmentation_workflow(...)` for the same
1. Logging and Checkpointing: 
	1. Log `dataset.csv` once to both CSVLogger and WandbLogger when beginning the experiment.
	1. Consider logging 
	2. Standardised metric names for `metrics.csv`, for ease of use for plotting metrics 
	3. ssh connector to download metrics to local files
	4. 