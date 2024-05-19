### TODO QoL
    Remove TQDM Progress Bars after run, set leave = true somehow

### TODO Tests
    Add tests for datasets and dataloaders, to ensure no surprises when passing data to model
    Add tests for plotting functions, with sample inputs
    
### Dataset Spec
1. Map Style Dataset:
    Attrs Interface:
        CLASS_NAMES: tuple[str,...], contains names of classes in the fixed order of encoding labels to integers
            e.g. "french_horn" encoded as 2 if CLASS_NAMES[2] == "french_horn"
        
        MEANS: tuple[float, ...], means for normalizing images
        STD_DEVS: tuple[float, ...], standard deviations for normalizing images

        df: DataFrame, represents the dataset (is logged for reproducibility)
            df.columns = image_path, mask_path/label_(str+idx), split
            all_paths in df should be string type

        split_df: DataFrame, split view of df
            split_df.columns = image_path, mask_path/label_idx, split, df_idx
            split_df["df_idx"] acts as a foreign key and references df.index
            all paths in split_df must be absolute paths
    
    Input Attrs: 
        root: Path,
        df: Optional[DataFrame], defines a custom train-val-test split for the dataset
        split: Literal["train", "val", "trainval", "test", "all"] 
        val_spit: float
        test_split: float
        random_seed: int
        image_transform: Optional[Transform] or DEFAULT_IMAGE_TRANSFORM
        target_transform: Optional[Transform] or DEFAULT_TARGET_TRANSFORM
        common_transform: Optional[Transform] or DEFAULT_COMMON_TRANSFORM
        
    Methods:
        __init__: None, obviously
        __len__: int, returns length of split_df
        __getitem__: 
            for classification: tuple[Tensor, int, int]
                image : Tensor.shape = (num_channels, height, width)
                label : int 
                df_idx : int
            for segmentation: tuple[Tensor, Tensor, int]
                image : Tensor.shape = (num_channels, height, width)
                mask : Tensor.shape = (num_classes, height, width), num_classes = 2 if binary
                df_idx : int
    
    ClassMethods:
        download_and_extract(url, root): None, to download that dataset from source
        classification_df or segmentation_df(random_seed, val_split, train_split, root): DataFrame
            to deterministically generate a dataset df with splits, called internally if no df provided
            can provide multiple such methods for different train-val-test split strategies
        write_to_streaming(remote, local): None, to write a streaming dataset, use StreamingDataset to load
        write_to_hdf(local): None, to write the dataset to hdf, use a modified df and HDFDataset to load

### Experiment Spec
    Use Pydantic and Custom Validation Logic to define an experiment 

    Experimental Parameters (as general as possible):

    Parameters
    __________
    dataset_name: name of the dataset, used to name the root directory 
    task: Literal["classification", "segmentation"], used to name bucket on the cloud
    random_seed: for deterministic experments, mandatory