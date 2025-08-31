### Optional Features 
- [] automate periodic sync of metrics, ckpt weights, inference results, etc. over SSH and S3
- [] cmd tool to plot live metrics from hdf (swmr)
- [x] switch to pixi with local venv
- [] Redirect [all] warnings to logfile

### Training Optmization Features
- [] add stocastic weight averaging
- [] add Inplace Activated Batch Norm to reduce memory usage
- [] Add options to change mode [fit, validate, test], select config.yaml, select profiling, delete logs dir before starting, to run.py
- [] Fix logger for DDP, something to do with computing torchmetrics on a single node 

### geovision.analysis/
- [] attribution Methods like Feature Visualization, Guided Backprop, IG, GRAD-CAM, LIME, SHAP, etc.
    -> [] (optionally) implement from scratch without using Captum 
- [] add color and class selectivity indices (read https://arxiv.org/pdf/1702.00382 and ConvNeXt v2)
- [] script to compute and store image statistics right in the metadata dataframe files, aggregate to compute dataset statistics
    -> load each image, calculate channel wise mean and variance and store in a dataframe -> concat with metadata to calculate aggregate stats dynamically for any split as properties
- [] classical statistics based EDA for datasets to highlight geometry and spectral characteristics

### geovision.data/
- [] add spatial sampler, at the image level (all images) and spatial level (georegistered scenes) [ref. Geo-Tiling for Semantic Segmentation]
    -> useful for inference time when entire images cannot be fed into the model
- [] add FMoW, BigEarthNet and HySpec11k (for pretraining)
- [] add PASTIS-R for agricultural analysis

- [] Inria: Buildings + (OSM) Road Centerlines
- [] City-OSM: Buildings + Roads
- [] RAMP: Buildings
- [] DeepGlobe Road Centerlines 
- [] HySpecNet-11k
- [] SpectralEarth 
- [] So2Sat-POP 
- [] Sen12Floods 

- ISPRS Vaihingen and Potsdam
- Massachusetts
- Kenya
- OpenCities
- Crowd Mapping Challenge
- LandCover.ai, Poland

- Spacenet AOI + Subset 
    - Rio: Buildings: Trash Labels
    - Vegas: Buildings + Roads
    - Paris: Buildings + Roads
    - Shanghai: Buildings + Roads
    - Khartoum: Buildings + Roads
    - Atlanta: Buildings (Off-Nadir)
    - Moscow: Roads
    - Mumbai: Roads
    - San Juan: Roads
    - Dar Es Salaam: Roads
    - Rotterdam: Buildings (MS+SAR)
    - SN7: Buildings (101 AOIs Worldwide * 24 Tiles (1 per month) PlanetLabs)
    - SN8: Flood Mapping (New Orleans, USA and Dernau, Germany)

#. UrbanVision:
    1. Building Segmentation (w/ Polygon Refinement).
    2. Road Segmentation (Centerline Extraction).
    3. Superresolution (Sentinel-2 or PlanetScope -> HighRes -> Polygons).
    4. Polygon Alignment (for Supervision from CrowdSourced / Noisy Data).
    5. Updating Maps over Time
    5. Data Fusion using SAR Imagery / DEMs / for additional supervision.
    6. Demonstrate Knowlege of Cadastal Mapping, Map Projections and Deep Learning.
    7. Performance in Densely Populated Locations.

#. AgriVision:
    1. Hyperspectral + SAR Foundation Model for Agriculture
    2. Crop Boundary Segmentation from High-Res Imagery
    3. Crop Type and Yield Estimation from Satellite Image Time Series
    4. Explain and Reason about Hyperspectral Model Predictions, Model Behaviour 
    5. Deployment on Edge Systems like Drones / Satellites (Smoll Model)