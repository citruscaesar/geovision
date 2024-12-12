from typing import Literal, Optional, Callable
from numpy.typing import NDArray

import h5py  # type: ignore
import torch
import zipfile
import tarfile
import shutil
import litdata
import warnings
import subprocess
import numpy as np
import pandas as pd
import pandera as pa
import imageio.v3 as iio
import torchvision.transforms.v2 as T  # type: ignore

from tqdm import tqdm
from io import BytesIO  #
from pathlib import Path
from itertools import chain
from litdata import optimize
from skimage.transform import resize
from torchvision.io import read_image, decode_jpeg, ImageReadMode
from multiprocessing import Pool, cpu_count

from geovision.data import Dataset, DatasetConfig
from geovision.io.remote import HTTPIO
from geovision.io.local import FileSystemIO as fs
#from geovision.data.interfaces import Dataset, DatasetConfig

import logging
logger = logging.getLogger(__name__)

# TODO:
# 1. find and download multilabel version for imagenet_1k
# 2. find and write the improved labels for imagenet_1k, imagenet_v2/v3
# 1. wordnet: dict[synset, list[class_name hierarchy]] for all 1000 classes, hardcoded into this file 
#   -> use to create 100 class multiclass classification subset
# 2. conceptualize, document and complete writing the extract, download, transform and load functions 

class ImagenetETL:
    local = Path.home() / "datasets" / "imagenet"

    # fmt: off
    imagenet_class_names =  (
        "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "cock",
        "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay",
        "magpie", "chickadee", "American dipper", "kite", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt",
        "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle",
        "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", "whiptail lizard", 
        "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon",
        "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake",
        "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "African rock python",
        "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder", "trilobite",
        "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow",
        "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peacock",
        "quail", "partridge", "grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill",
        "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus",
        "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug",
        "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster",
        "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret",
        "bittern", "crane (bird)", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank",
        "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion",
        "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier",
        "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound",
        "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound",
        "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier",
        "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier",
        "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier",
        "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer",
        "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier",
        "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever",
        "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter",
        "Brittany Spaniel", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniels", "Sussex Spaniel",
        "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael", "Malinois", "Briard", "Australian Kelpie", "Komondor",
        "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier desFlandres", "Rottweiler", "German Shepherd Dog",
        "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund",
        "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky",
        "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland",
        "Pyrenean Mountain Dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "Griffon Bruxellois", "Pembroke Welsh Corgi",
        "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog", "grey wolf",
        "Alaskan tundra wolf", "red wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox",
        "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard",
        "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", 
        "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle",
        "weevil", "fly", "bee", "ant", "grasshopper", "cricket", "stick insect", "cockroach", "mantis", "cicada", "leafhopper",
        "lacewing", "dragonfly", "damselfly", "red admiral", "ringlet", "monarch butterfly", "small white", "sulphur butterfly",
        "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit",
        "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel", "zebra", "pig", "wild boar",
        "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram", "bighorn sheep", "Alpine ibex", "hartebeest", "impala",
        "gazelle", "dromedary", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger",
        "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon",
        "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi",
        "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant",
        "red panda", "giant panda", "snoek", "eel", "coho salmon", "rock beauty", "clownfish", "sturgeon", "garfish", "lionfish",
        "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship",
        "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "waste container", "assault rifle", "backpack",
        "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster", "barbell", "barber chair", "barbershop",
        "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel",
        "bathtub", "station wagon", "lighthouse", "beaker", "military cap", "beer bottle", "beer glass", "bell-cot", "bib", "tandem bicycle",
        "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore",
        "bottle cap", "bow", "bow tie", "brass", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest",
        "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror",
        "carousel", "tool kit", "carton", "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran",
        "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "chest", "chiffonier", "chime",
        "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker",
        "coffee mug", "coffeemaker", "coil", "combination lock", "computer keyboard", "confectionery store", "container ship", "convertible",
        "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "crane (machine)", "crash helmet", "crate", "infant bed", "Crock Pot",
        "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock",
        "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig",
        "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center",
        "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire engine", "fire screen sheet",
        "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car",
        "French horn", "frying pan", "fur coat", "garbage truck", "gas mask", "gas pump", "goblet", "go-kart", "golf ball", "golf cart",
        "gondola", "gong", "gown", "grand piano", "greenhouse", "grille", "grocery store", "guillotine", "barrette", "hair spray", "half-track",
        "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "harvester",
        "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt", "horizontal bar", "horse-drawn vehicle", "hourglass",
        "iPod", "clothes iron", "jack-o'-lantern", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "pulled rickshaw", "joystick", "kimono",
        "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "paper knife", "library", "lifeboat",
        "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "speaker", "loupe", "sawmill", "magnetic compass",
        "mail bag", "mailbox", "tights", "tank suit", "manhole cover", "maraca", "marimba", "mask", "match", "maypole", "maze",
        "measuring cup", "medicine chest", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt",
        "minivan", "missile", "mitten", "mixing bowl", "mobile home", "Model T", "modem", "monastery", "monitor", "moped", "mortar",
        "square academic cap", "mosque", "mosquito net", "scooter", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "nail",
        "neck brace", "necklace", "nipple", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "organ",
        "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "packet", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas",
        "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "passenger car", "patio",
        "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube",
        "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship",
        "pitcher", "hand plane", "planetarium", "plastic bag", "plate rack", "plow", "plunger", "Polaroid camera", "pole", "police van",
        "poncho", "billiard table", "soda bottle", "pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison",
        "projectile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio",
        "radio telescope", "rain barrel", "recreational vehicle", "reel", "reflex camera", "refrigerator", "remote control", "restaurant",
        "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler", "running shoe", "safe", "safety pin", "salt shaker",
        "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT screen", "screw", 
        "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji", "shopping basket", "shopping cart", "shovel", "shower cap",
        "shower curtain", "ski", "ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow",
        "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "space bar", "space heater", "space shuttle",
        "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum",
        "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit",
        "sundial", "sunglass", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", "swimsuit", "swing", "switch", "syringe", "table lamp",
        "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine",
        "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck",
        "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "tub", "turnstile", "typewriter keyboard",
        "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vault", "velvet", "vending machine", "vestment", "viaduct", "violin",
        "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug",
        "water tower", "whiskey jug", "whistle", "wig", "window screen", "window shade", "Windsor tie", "wine bottle", "wing", "wok", "wooden spoon",
        "wool", "split-rail fence", "shipwreck", "yawl", "yurt", "website", "comic book", "crossword", "traffic sign", "traffic light", "dust jacket",
        "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "ice pop", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog",
        "mashed potato", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke",
        "bell pepper", "cardoon", "mushroom", "Granny Smith", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "custard apple",
        "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "cup", "eggnog",
        "alp", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "shoal", "seashore", "valley", "volcano", "baseball player", 
        "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", 
        "agaric", "gyromitra", "stinkhorn mushroom", "earth star", "hen-of-the-woods", "bolete", "ear of corn", "toilet paper"
    )

    imagenette_class_names = ( 
        "tench", "english_springer", "cassette_player", "chain_saw", "church", "french_horn", "garbage_truck", "gas_pump", "golf_ball", "parachute"
    )

    warning_idxs = (
        14447, 55541, 59005, 82043, 109039, 132926, 147946, 148200, 148461, 149091, 149075, 149075, 149091, 149150, 149157, 149175, 149179, 149210, 161793,
        164725, 170021, 170122, 197088, 251657, 258309, 314614, 389572, 426998, 434647, 455142, 456227, 476489, 498503, 499108, 499266, 507884, 526417,
        551234, 574523, 626422, 696344, 724759, 752846, 787840, 793951, 793981, 801232, 802987, 816270, 817533, 821501, 831922, 841920, 876793, 884208, 
        888236, 897070, 943737, 1036821, 1038877, 1042938, 1067042, 1095673, 1139200, 1190161, 1290684, 1312510, 1318704, 1319836 
    )
    # fmt: on

    means = (0.485, 0.456, 0.406)
    std_devs = (0.229, 0.224, 0.225)
    imagenet_default_config = DatasetConfig(
        random_seed=42,
        tabular_sampler_name="imagefolder_notest",
        tabular_sampler_params={"val_frac": 0.05},
        image_pre=T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Resize(224, antialias=True),
        ]),
        target_pre=T.Identity(),
        train_aug=T.RandomHorizontalFlip(0.5),
        eval_aug=T.Identity(),
    )

    @classmethod
    def extract(cls, src: Literal["kaggle", "huggingface", "fastai"], subset: Literal["imagenet_1k", "imagenette"]):
        """extract, preprocess and save imagenet-1k archive to :imagenet/archives/imagenet-object-localization-challenge.zip"""
        assert subset in ("imagenet_1k", "imagenette")
        if subset == "imagenet_1k":
            assert src in ("kaggle", "huggingface")
            if src == "kaggle":
                # download to cls.local / "staging" 
                subprocess.call(["kaggle competitions download -c imagenet-object-localization-challenge"])
            elif src == "huggingface":
                # download multipart from ILSVRC/imagenet-1k and join them as a single archive (symlinked or sth?)
                ...
        elif subset == "imagenette":
            assert src in ("fastai", )
            if src == "fastai":
                HTTPIO.download_url("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz", cls.local / "staging")

    @classmethod
    def download(cls, subset: Literal["imagenet_1k", "imagenet_100", "imagenette"]):
        assert subset in ("imagenet_1k", "imagenet_100", "imagenette") 

    @classmethod
    def load(
        cls, 
        table: Literal["index", "corrupt", "synsets"], 
        src: Literal["archive", "imagefolder", "hdf5"], 
        subset: Literal["imagenet_1k", "imagenet_100", "imagnette"],
    ) -> pd.DataFrame:

        assert src in ("archive", "imagefolder", "hdf5")
        assert subset in ("imagenet_1k", "imagenet_100", "imagenette")
        if subset == "imagenet_1k":
            def _assign_labels(df: pd.DataFrame):
                synsets = sorted(df["label_synset"].unique())
                return (
                    df.sort_values("label_synset")
                    .assign(label_idx=lambda df: df["label_synset"].apply(lambda x: synsets.index(x)))
                    .assign(label_str=lambda df: df["label_idx"].apply(lambda x: cls.imagenet_class_names[x]))
                    .sort_values("label_idx")
                    .reset_index(drop=True)
                )
            if src == "archive":
                archive_path = fs.get_valid_file_err(cls.local, "staging", "imagenet-object-localization-challenge.zip")
                metadata_path = archive_path.parent / "imagenet_1k_metadata.h5"
                try:
                    return pd.read_hdf(metadata_path, key = table, mode = 'r')
                except OSError:
                    assert table == "index", f"expected :table to be 'index' since {table} metadata was not found and cannot be computed by this fn"

                    with zipfile.ZipFile(archive_path) as zf:
                        zip_images = zf.namelist()  
                        val_df = pd.read_csv(zf.open("LOC_val_solution.csv"))

                    train_df = (
                        pd.DataFrame({"image_path": [n for n in zip_images if n.endswith(".JPEG") and "train" in n]}).
                        assign(label_synset=lambda df: df["image_path"].apply(lambda x: x.split('/')[-2]))
                    )
                    val_df = (
                        val_df
                        .assign(image_path=lambda df: df["ImageId"].apply(lambda x: str(Path("ILSVRC")/"Data"/"CLS-LOC"/"val"/f"{x}.JPEG")))
                        .assign(label_synset=lambda df: df["PredictionString"].apply(lambda x: x.split(" ")[0]))
                        .drop(columns=["ImageId", "PredictionString"])
                    )
                    df = pd.concat([train_df, val_df]).pipe(_assign_labels)
                    df.to_hdf(metadata_path, key = "index", mode = "a") # NOTE: don't want to remove any other metadata from the file, mode = a
                    return df

            elif src == "imagefolder":
                imagefolder_path = fs.get_valid_dir_err(cls.local / "imagefolder" / subset, empty_ok=False)
                metadata_path = imagefolder_path / "metadata.h5"
                try:
                    return pd.read_hdf(metadata_path, key = table, mode = 'r')
                except OSError:
                    assert table == "index", f"expected :table to be 'index' since {table} metadata was not found and cannot be computed by this fn"
                    df = (
                        pd.DataFrame({"image_path": list(chain((imagefolder_path / "train").rglob("*.jpg"), (imagefolder_path / "val").rglob("*.jpg")))})
                        .assign(image_path=lambda df: df["image_path"].apply(lambda x: '/'.join(str(x).split('/')[-3:])))
                        .assign(label_synset=lambda df: df["image_path"].apply(lambda x: x.split('/')[-2]))
                        .pipe(_assign_labels)
                    )
                    return df

            elif src == "hdf5":
                return pd.read_hdf(cls.local / "hdf5" / "imagenet_1k.h5", key = table, mode = 'r') 
        
        elif subset == "imagenet_100":
            raise NotImplementedError 

        elif subset == "imagenette":
            def _assign_labels(df: pd.DataFrame) -> pd.DataFrame:
                class_synsets = sorted(df["image_path"].apply(lambda x: x.split('/')[-2]).unique())
                return (
                    df
                    .assign(label_str=lambda df: df["image_path"].apply(lambda x: x.split('/')[-2]))
                    .assign(label_idx=lambda df: df["label_str"].apply(lambda x: class_synsets.index(x)))
                    .assign(label_str=lambda df: df["label_idx"].apply(lambda x: cls.imagenette_class_names[x]))
                    .sort_values("label_idx")
                    .reset_index(drop=True)
                )
            if src == "archive":
                archive_path = fs.get_valid_file_err(cls.local, "staging", "imagenette2.tgz")
                metadata_path = archive_path.parent / "imagenette_metadata.h5"
                try: 
                    return pd.read_hdf(metadata_path, key = table, mode = 'r')
                except OSError:
                    with tarfile.open(archive_path) as a:
                        df = pd.DataFrame({"image_path": [p for p in a.getnames() if p.endswith(".JPEG")]})
                    df = df.pipe(_assign_labels)
                    df.to_hdf(metadata_path, key = "index", mode = 'w')
                    return df

            elif src == "imagefolder":
                imagefolder_path = fs.get_valid_dir_err(cls.local, "imagefolder", subset)
                metadata_path = imagefolder_path.parent / "metadata.h5"
                try:
                    return pd.read_hdf(metadata_path, key = table, mode = 'r')
                except OSError:
                    df = (
                        pd.DataFrame({"image_path": list(imagefolder_path.rglob("*.jpg"))})
                        .assign(image_path = lambda df: df["image_path"].apply(lambda x: '/'.join(x.split('/')[-3:])))
                        .pipe(_assign_labels)
                    )
                    df.to_hdf(metadata_path, key = "index", mode = 'a')
                    return df

            elif src == "hdf5":
                return pd.read_hdf(cls.local / "hdf5" / "imagenette.h5", key = table, mode = 'r') 

    @classmethod
    def transform(
        cls, 
        to: Literal["hdf5", "imagefolder"], 
        subset: Literal["imagenet_1k", "imagenet_100", "imagenette"], 
        df_transform: Optional[Callable[..., pd.DataFrame] | list[Callable[..., pd.DataFrame]]] = None,
        resize_to: int = 0, 
        jpg_quality: int = 95,
        chunks: int = 0,
        num_parts: int = 15,
        num_proc: Optional[int] = None,
        val_only: bool = False,
        **kwargs
    ):
        assert to in ("hdf5", "imagefolder", "litdata")
        assert subset in ("imagenet_1k", "imagenette")
        if subset == "imagenet_1k":
            archive_path = cls.local / "staging" / "imagenet-object-localization-challenge.zip"
            assert fs.is_valid_file(archive_path)

            if to == "imagefolder":
                assert not val_only, ":val_only not implemented for imagefolder"
                imagefolder_path = fs.get_new_dir(cls.local / to / subset)
                temp_path = fs.get_new_dir(cls.local / "temp")

                df = cls.load(table = 'index', src = 'archive', subset = 'imagenet_1k')
                if df_transform is not None:
                    if isinstance(df_transform, list):
                        for transform in df_transform:
                            df = df.pipe(transform)
                    elif callable(df_transform):
                        df = df.pipe(df_transform)
                    else:
                        raise AssertionError("expected :df_transform to be a fn: df -> df, or a list of such fn")

                # extract archive contents to temp
                # TODO: add resize, re-encoding and progressbar
                with zipfile.ZipFile(archive_path) as zf:
                    zf.extractall(temp_path, df["image_path"])

                # move metadata files to imagefolder
                shutil.move(temp_path / "LOC_synset_mapping.txt", imagefolder_path / "LOC_synset_mapping.txt")
                shutil.move(temp_path / "LOC_val_solution.csv", imagefolder_path / "LOC_val_solution.csv")

                # prepare subdirs for imagefolder
                temp_train = temp_path / "ILSVRC" / "Data" / "CLS-LOC" / "train"
                for label_synset in df["label_synset"].unique():
                    fs.get_new_dir(imagefolder_path, "train", label_synset)
                    fs.get_new_dir(imagefolder_path, "val", label_synset)
                test_dir = fs.get_new_dir(imagefolder_path, "test")

                # move train
                for temp_path in temp_train.rglob("*.JPEG"):
                    image_path = imagefolder_path / "train" / temp_path.parent.name / f"{temp_path.stem}.jpg"
                    shutil.move(temp_path, image_path)

                # move val
                val_df = (
                    pd.read_csv(imagefolder_path / "LOC_val_solution.csv")
                    .assign(temp_path=lambda df: df["ImageId"].apply(lambda x: temp_path / "ILSVRC" / "Data" / "CLS-LOC" / "val" / f"{x}.JPEG"))
                    .assign(parent_dir=lambda df: df["PredictionString"].apply(lambda x: x.split(" ")[0]))
                    .assign(image_path=lambda df: df.apply(lambda x: imagefolder_path / "val" / x["parent_dir"] / f"{x["ImageId"]}.jpg", axis=1))
                    .drop(columns=["ImageId", "PredictionString"])
                )
                for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc=f"{imagefolder_path / "val"}"):
                    shutil.copy(row["temp_path"], row["image_path"])

                # move test
                for temp_path in tqdm(list((temp_path/"ILSVRC"/"Data"/"CLS-LOC"/"test").rglob("*.JPEG")), desc=f"{imagefolder_path / "test"}"):
                    shutil.copy(temp_path, test_dir/f"{temp_path.stem}.jpg")

                # cleanup
                shutil.rmtree(temp_path)
                cls.load('index', 'imagefolder', 'imagenet_1k')

            elif to == "hdf5":
                assert resize_to >= 0 
                assert jpg_quality >= 0 and jpg_quality <= 95 
                assert chunks >= 0
                assert num_parts >= 1

                hdf5_path = fs.get_new_dir(cls.local, "hdf5") / "imagenet_1k.h5"
                df = cls.load('index', 'archive', subset)#.assign(image_path = lambda df: df["image_path"].apply(lambda x: x.as_posix()))

                if df_transform is not None:
                    if isinstance(df_transform, list):
                        for transform in df_transform:
                            df = df.pipe(transform)
                    elif callable(df_transform):
                        df = df.pipe(df_transform)
                    else:
                        raise AssertionError("expected :df_transform to be a fn: df -> df, or a list of such fn")
 
                if val_only:
                    num_parts = 1
                    hdf5_path = hdf5_path.parent / "imagenet_1k_val.h5"
                    df = df.loc[lambda df: df["image_path"].apply(lambda x: "val" in x)]

                if num_parts == 1:                
                    cls._write_to_hdf(df, archive_path, hdf5_path, resize_to, jpg_quality, chunks)

                else:
                    args = list()
                    for i, idxs in enumerate(np.array_split(df.index, num_parts)):
                        args.append((df.iloc[idxs], archive_path, hdf5_path.parent/f"{hdf5_path.stem}_part={i}.h5", resize_to, jpg_quality, chunks))
                    
                    num_proc = num_proc or cpu_count() - 1
                    with Pool(num_proc) as pool:
                        pool.starmap(cls._write_to_hdf, args)

                    with h5py.File(hdf5_path, mode='w') as f:
                        vlen_dtype = h5py.special_dtype(vlen = np.uint8)
                        layout = h5py.VirtualLayout(shape = len(df), dtype = vlen_dtype)
                        for idx, args in enumerate(args):
                            vds_df, vds_path = args[0], args[2]
                            layout[idx*len(vds_df): (idx+1)*len(vds_df)] = h5py.VirtualSource(vds_path, "images", len(df), vlen_dtype)
                        f.create_virtual_dataset("images", layout)

                    df["image_path"] = df["image_path"].apply(lambda x: '/'.join(x.split('/')[-3:]))
                    df.to_hdf(hdf5_path, mode = 'r+', key = "index")
            
            elif to == "litdata":
                df = cls.load('index', 'archive', subset).iloc[:10000]
                optimize(
                    fn=cls._write_to_litdata,
                    inputs=[(idx, row, archive_path, resize_to, jpg_quality) for idx, row in df.iterrows()],
                    output_dir=str(fs.get_new_dir(cls.local / "litdata")),
                    chunk_bytes="512MB",
                    num_workers=num_proc
                )

        elif subset == "imagenette":
            archive_path = fs.get_valid_file_err(cls.local, "staging", "imagenette2.tgz")

            if to == "imagefolder":
                imagefolder_path = fs.get_new_dir(cls.local, "imagefolder")
                temp_path = fs.get_new_dir(cls.local, "temp")
                with tarfile.open(archive_path) as tf:
                    tf.extractall(temp_path)

                def move_tempfiles_to_imagefolder(split: Literal["train", "val"]):
                    for src_path in tqdm(list((temp_path / "imagenette2" / split).rglob("*.JPEG")), desc=f"{imagefolder_path / split}"):
                        dst_path = imagefolder_path / split / src_path.parent.stem / f"{src_path.stem}.jpg"
                        dst_path.parent.mkdir(exist_ok=True, parents=True)
                        shutil.move(src_path, dst_path)

                move_tempfiles_to_imagefolder("train")
                move_tempfiles_to_imagefolder("val")
                shutil.rmtree(temp_path)
                cls.load('index', 'imagefolder', 'imagenette')

            elif to == "hdf5":
                hdf5_path = fs.get_new_dir(cls.local, "hdf5") / "imagenette.h5"
                df = cls.load('index', 'archive', 'imagenette')

                with tarfile.open(archive_path, mode = 'r') as tf:
                    with h5py.File(hdf5_path, mode = 'w') as f: 
                        images = f.create_dataset("images", len(df), h5py.special_dtype(vlen = np.uint8), chunks = chunks)
                        for idx, row in tqdm(df.iterrows(), total = len(df)):
                            image = iio.imread(tf.extractfile(row["image_path"])).squeeze()
                            images[idx] = cls._write_to_jpg(image, resize_to, jpg_quality)

                df["image_path"] = df["image_path"].apply(lambda x: x.split('/')[-3:])
                df.to_hdf(hdf5_path, key = 'index', mode = 'r+')

    @classmethod
    def _write_to_hdf(cls, df: pd.DataFrame, archive_path: Path, ds_path: Path, resize_to: int, jpg_quality: int, chunks: int):
        with zipfile.ZipFile(archive_path, mode = 'r') as zf:
            with h5py.File(ds_path, mode = 'w') as f:
                images = f.create_dataset("images", shape = len(df), dtype = h5py.special_dtype(vlen = np.uint8), chunks = chunks) 
                for idx, row in tqdm(df.reset_index().iterrows(), total = len(df)):
                    image = iio.imread(zf.read(row["image_path"])).squeeze()
                    images[idx] = cls._write_to_jpg(image, resize_to, jpg_quality)

        df["image_path"] = df["image_path"].apply(lambda x: x.split('/')[-3:])
        df.to_hdf(ds_path, key = "index", mode = 'r+')

    @classmethod
    def _write_to_litdata(cls, args: tuple[int, pd.Series, Path, int, int]):
    #def _write_to_litdata(cls, row: pd.Series, zf: zipfile.ZipFile, resize_to: int, jpg_quality: int):
        idx, row, archive_path, resize_to, jpg_quality = args
        return {
            "image": cls._write_to_jpg(iio.imread(archive_path/row["image_path"]), resize_to, jpg_quality),
            "label_idx": row["label_idx"],
            "df_idx": idx 
        }

    @staticmethod
    def _write_to_jpg(image: NDArray, resize_to:int, jpg_quality: int) -> NDArray:
        if resize_to:
            image = resize(image, (resize_to, resize_to), preserve_range=True, anti_aliasing=True).astype(np.uint8)
        if image.ndim == 2: # greyscale images -> repeat greyscale image across channels
            image = np.stack((image,)*3, axis = -1)
        elif image.shape[-1] == 4: # rgba images -> remove alpha (transparency) channel
            image = image[:, :, :3]
        return np.frombuffer(iio.imwrite("<bytes>", image, extension=".jpg", quality = jpg_quality), dtype = np.uint8)

    
    @staticmethod
    def _list_corrupt_images(df: pd.DataFrame, archive_path: Path):
        corrupted = list()
        with zipfile.ZipFile(archive_path) as zf:
            for idx, row in tqdm(df.iterrows(), total = len(df)):
                try:
                    iio.imread(zf.read(row["image_path"]), extension='.jpeg')
                except Warning as w:
                    corrupted.append(idx)
                    print(f"warning on idx = {idx}, [{w}]")
                    continue
                except Exception as e:
                    corrupted.append(idx)
                    print(f"exception on idx = {idx}, [{e}]")
                    continue
        return corrupted
    
    @classmethod
    def get_corrupt_images_df(cls) -> pd.DataFrame:
        archive_path = cls.local / "staging" / "imagenet-object-localization-challenge.zip"
        df = cls.load(table = 'index', src = 'archive', subset = 'imagenet_1k')

        # TODO: setup logging and write warnings/errors to a file instead of stdout and memory
        warnings.filterwarnings('error')
        with Pool(cpu_count()-1) as pool:
            sus_idxs = pool.starmap(cls._list_corrupt_images, [(df.loc[idxs], archive_path) for idxs in np.array_split(df.index, cpu_count()-1)])
        warnings.resetwarnings()

        idxs = list() 
        for subarray in sus_idxs:
            for idx in subarray:
                idxs.append(idx)
        df = df.loc[idxs]
        df.to_hdf(archive_path.parent / "metadata.h5", key = "corrupted", mode = "a")
        return df

    @classmethod
    def filter_corrupt_images(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(index=cls.load(table = 'corrupt', src = 'archive', subset = 'imagenet_1k').index)

    @classmethod
    def get_dataset_df_from_litdata(cls, config: Optional[DatasetConfig] = None, schema: Optional[pa.DataFrameSchema] = None) -> pd.DataFrame:
        try:
            return pd.read_csv(cls.local/"litdata"/"dataset.csv")
        except OSError:
            if config is None:
                raise ValueError("since dataset df not found at :imagenet/litdata, config cannot be none")
            return DatasetConfig.get_df(config, schema, cls.get_dataset_df_from_imagefolder().reset_index(drop=False, names="df_idx")).assign(
                image_path=lambda df: df["image_path"].apply(lambda x: cls.local / "imagefolder" / Path(x))
            )
            # NOTE: don't save df to litdata dir here, it is saved by worker 0 in the litdata encoder script

Imagenet1KIndexSchema = pa.DataFrameSchema(
    columns={
        "image_path": pa.Column(str, coerce=True),
        "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, 1000)))),
        # TODO:
        # check label synset is in wordnet 
        # check label str is one of the correct corresponding classnames
        # "label_synset": pa.Column(str, pa.Check.isin(ImagenetETL.wordnet.keys())),
        "label_str": pa.Column(str, pa.Check.isin(ImagenetETL.imagenet_class_names)),
        "split": pa.Column(str, pa.Check.isin(Dataset.valid_splits))
    }, 
    index=pa.Index(int, unique=True),
)

ImagenetteIndexSchema = pa.DataFrameSchema(
    columns={
        "image_path": pa.Column(str, coerce=True),
        "label_idx": pa.Column(int, pa.Check.isin(range(0, 10))),
        "label_str": pa.Column(str, pa.Check.isin(ImagenetETL.imagenette_class_names)),
        "split": pa.Column(str, pa.Check.isin(Dataset.valid_splits))
    }, 
    index=pa.Index(int, unique=True),
)

class ImagenetImagefolderClassification(Dataset):
    name = "imagenet_1k"
    task = "classification"
    subtask = "multiclass"
    storage = "imagefolder"
    class_names = ImagenetETL.imagenet_class_names
    num_classes = 1000 
    root = ImagenetETL.local/"imagefolder"/"imagenet_1k"
    schema = Imagenet1KIndexSchema
    config = ImagenetETL.imagenet_default_config
    loader = ImagenetETL.load

    def __init__(self, split: Literal["train", "val", "test", "trainvaltest", "all"] = "all", config: Optional[DatasetConfig] = None):
        super().__init__(split, config)
        self.df = self.get_df(split, prefix_root_to_paths=True)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        idx_row = self.df.iloc[idx]
        image = read_image(idx_row["image_path"], mode=ImageReadMode.RGB)
        image = self.config.image_pre(image)
        if self.split in ("train", "trainvaltest"):
            image = self.config.train_aug(image)
        elif self.split in ("val", "test"):
            image = self.config.eval_aug(image)
        return image, idx_row["label_idx"], idx_row["df_idx"]

class ImagenetHDF5Classification(Dataset):
    name = "imagenet_1k"
    task = "classification"
    subtask = "multiclass"
    storage = "hdf5"
    class_names = ImagenetETL.imagenet_class_names
    num_classes = 1000 
    root = ImagenetETL.local/"hdf5"/"imagenet_1k.h5"
    schema = Imagenet1KIndexSchema
    config = ImagenetETL.imagenet_default_config
    loader = ImagenetETL.load

    def __init__(self, split: Literal["train", "val", "test", "trainvaltest", "all"] = "all", config: Optional[DatasetConfig] = None):
        super().__init__(split, config)
        self.df = self.get_df(split, prefix_root_to_paths=False)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        idx_row = self.df.iloc[idx]
        with h5py.File(self.root, mode="r") as hdf5_file:
            # image = iio.imread(BytesIO(hdf5_file["images"][idx_row["df_idx"]]))
            image = decode_jpeg(torch.tensor(hdf5_file["images"][idx_row["df_idx"]], dtype = torch.uint8), ImageReadMode.RGB)
        image = self._config.image_pre(image)
        if self._split in ("train", "trainvaltest"):
            image = self._config.train_aug(image)
        elif self._split in ("val", "test"):
            image = self._config.eval_aug(image)
        return image, idx_row["label_idx"], idx_row["df_idx"]

class ImagenetteImagefolderClassification(Dataset):
    name = "imagenette"
    task = "classification"
    subtask = "multiclass"
    storage = "imagefolder"
    class_names = ImagenetETL.imagenette_class_names
    num_classes = 10 
    root = ImagenetETL.local/"imagefolder"/"imagenette"
    schema = ImagenetteIndexSchema
    config = ImagenetETL.imagenet_default_config
    loader = ImagenetETL.load

    def __init__(self, split: Literal["train", "val", "test", "trainvaltest", "all"] = "all", config: Optional[DatasetConfig] = None):
        super().__init__(split, config)
        self.df = self.get_df(split, prefix_root_to_paths=True)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        idx_row = self.df.iloc[idx]
        image = read_image(idx_row["image_path"], mode=ImageReadMode.RGB)
        image = self.config.image_pre(image)
        if self.split in ("train", "trainvaltest"):
            image = self.config.train_aug(image)
        elif self.split in ("val", "test"):
            image = self.config.eval_aug(image)
        return image, idx_row["label_idx"], idx_row["df_idx"]

class ImagenetteHDF5Classification(Dataset):
    name = "imagenette"
    task = "classification"
    subtask = "multiclass"
    storage = "hdf5"
    class_names = ImagenetETL.imagenette_class_names
    num_classes = 10
    root = ImagenetETL.local/"hdf5"/"imagenette.h5"
    schema = ImagenetteIndexSchema
    config = ImagenetETL.imagenet_default_config
    loader = ImagenetETL.load

    def __init__(self, split: Literal["train", "val", "test", "trainvaltest", "all"] = "all", config: Optional[DatasetConfig] = None):
        super().__init__(split, config)
        self.df = self.get_df(split, prefix_root_to_paths=False)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        idx_row = self.df.iloc[idx]
        with h5py.File(self.root, mode="r") as hdf5_file:
            image = iio.imread(BytesIO(hdf5_file["images"][idx_row["df_idx"]]))
            #image = decode_jpeg(torch.tensor(hdf5_file["images"][idx_row["df_idx"]], dtype = torch.uint8), ImageReadMode.RGB)
        image = self._config.image_pre(image)
        if self._split in ("train", "trainvaltest"):
            image = self._config.train_aug(image)
        elif self._split in ("val", "test"):
            image = self._config.eval_aug(image)
        return image, idx_row["label_idx"], idx_row["df_idx"]

# class ImagenetLitDataClassification(litdata.StreamingDataset, Dataset):
    # name = "imagenet_litdata_classification"
    # class_names = ImagenetETL.imagenet_class_names
    # num_classes = len(ImagenetETL.imagenet_class_names)
    # means = ImagenetETL.means
    # std_devs = ImagenetETL.std_devs
    # df_schema = pa.DataFrameSchema(
        # {
            # "image_path": pa.Column(str, coerce=True),
            # "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, len(ImagenetETL.imagenet_class_names))))),
            # "label_str": pa.Column(str),
            # "split": pa.Column(str, pa.Check.isin(Dataset.splits)),
        # },
        # index=pa.Index(int, unique=True),
    # )

    # split_df_schema = pa.DataFrameSchema(
        # {
            # "image_path": pa.Column(str, coerce=True),
            # "df_idx": pa.Column(int),
            # "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, len(ImagenetETL.imagenet_class_names))))),
        # },
        # index=pa.Index(int, unique=True),
    # )

    # def __init__(self, split: Literal["train", "val", "test"] = "train", config: Optional[DatasetConfig] = None):
        # assert split in ("train", "val", "test"), f"invalid :split, got {split}"
        # self._root = fs.get_valid_dir_err(ImagenetETL.local, "litdata", split)
        # self._split = self.get_valid_split_err(split)
        # self._config = config or ImagenetETL.default_config
        # self._df = self.df_schema(ImagenetETL.get_dataset_df_from_litdata())
        # self._split_df = self._config.verify_and_get_split_df(df=self._df, schema=self.split_df_schema, split=self._split)
        # super().__init__(input_dir=str(self._root), shuffle=True if split == "train" else False, drop_last=False, seed=config.random_seed, max_cache_size="200GB")

    # def __getitem__(self, idx: int):
        # data = super().__getitem__(idx)
        # image = self._config.image_pre(data["image"])
        # if self.split in ("train", "trainvaltest"):
            # image = self._config.train_aug(image)
        # elif self._split in ("val", "test"):
            # image = self._config.eval_aug(image)
        # return image, data["label_idx"], data["df_idx"]

    # @property
    # def root(self):
        # return self._root

    # @property
    # def split(self) -> Literal["train", "val", "test"]:
        # return self._split

    # @property
    # def df(self) -> pd.DataFrame:
        # return self._df

    # @property
    # def split_df(self) -> pd.DataFrame:
        # return self._split_df

# class ImagenetteImagefolderClassification(Dataset):
    # name = "imagenette_imagefolder_classification"
    # class_names = ImagenetETL.imagenette_class_names
    # num_classes = len(ImagenetETL.imagenette_class_names)
    # means = ImagenetETL.means
    # std_devs = ImagenetETL.std_devs
    # df_schema = pa.DataFrameSchema(
        # {
            # "image_path": pa.Column(str, coerce=True),
            # "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, len(ImagenetETL.imagenette_class_names))))),
            # "label_str": pa.Column(str, pa.Check.isin(ImagenetETL.imagenette_class_names)),
            # "split": pa.Column(str, pa.Check.isin(Dataset.splits)),
        # },
        # index=pa.Index(int, unique=True),
    # )

    # split_df_schema = pa.DataFrameSchema(
        # {
            # "image_path": pa.Column(str, coerce=True),
            # "df_idx": pa.Column(int, unique=True),
            # "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, len(ImagenetETL.imagenette_class_names))))),
        # },
        # index=pa.Index(int, unique=True),
    # )

    # def __init__(self, split: Literal["train", "val", "test", "trainvaltest", "all"] = "all", config: Optional[DatasetConfig] = None) -> None:
        # self._root = fs.get_valid_dir_err(ImagenetETL.local / "imagefolder" / "imagenette")
        # self._split = self.get_valid_split_err(split)
        # self._config = config or ImagenetETL.default_config
        # logger.info(
            # f"init {self.name}[{self._split}]\nimage_pre = {self._config.image_pre}\ntarget_pre = {self._config.target_pre}\ntrain_aug = {self._config.train_aug}\neval_aug = {self._config.eval_aug}"
        # )
        # self._df = self._config.verify_and_get_df(schema=self.df_schema, fallback_df=ImagenetETL.load('index', 'imagefolder', 'imagenette'))
        # self._split_df = self._config.verify_and_get_split_df(df=self._df, schema=self.split_df_schema, split=self._split)

    # def __len__(self):
        # return len(self._split_df)

    # def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        # idx_row = self._split_df.iloc[idx]
        # image = iio.imread(idx_row["image_path"], format_hint=".jpg").squeeze()
        # image = np.stack((image,) * 3, axis=-1) if image.ndim == 2 else image
        # image = self._config.image_pre(image)
        # if self.split in ("train", "trainvaltest"):
            # image = self._config.train_aug(image)
        # elif self._split in ("val", "test"):
            # image = self._config.eval_aug(image)
        # return image, idx_row["label_idx"], idx_row["df_idx"]

    # @property
    # def root(self):
        # return self._root

    # @property
    # def split(self) -> Literal["train", "val", "test", "trainvaltest", "all"]:
        # return self._split

    # @property
    # def df(self) -> pd.DataFrame:
        # return self._df

    # @property
    # def split_df(self) -> pd.DataFrame:
        # return self._split_df

# class ImagenetteHDF5Classification(Dataset):
    # name = "imagenette_hdf5_classification"
    # class_names = ImagenetETL.imagenette_class_names
    # num_classes = len(ImagenetETL.imagenette_class_names)
    # means = ImagenetETL.means
    # std_devs = ImagenetETL.std_devs
    # df_schema = pa.DataFrameSchema(
        # {
            # "image_path": pa.Column(str, coerce=True),
            # "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, len(ImagenetETL.imagenette_class_names))))),
            # "label_str": pa.Column(str),
            # "split": pa.Column(str, pa.Check.isin(Dataset.splits)),
        # },
        # index=pa.Index(int, unique=True),
    # )

    # split_df_schema = pa.DataFrameSchema(
        # {
            # "image_path": pa.Column(str, coerce=True),
            # "df_idx": pa.Column(int),
            # "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, len(ImagenetETL.imagenette_class_names))))),
        # },
        # index=pa.Index(int, unique=True),
    # )

    # def __init__(self, split: Literal["train", "val", "test", "trainvaltest", "all"] = "all", config: Optional[DatasetConfig] = None) -> None:
        # self._root = fs.get_valid_file_err(ImagenetETL.local, "hdf5", "imagenette.h5")
        # self._split = self.get_valid_split_err(split)
        # self._config = config or ImagenetETL.default_config
        # logger.info(
            # f"init {self.name}[{self._split}] using\nimage_pre = {self._config.image_pre}\ntarget_pre = {self._config.target_pre}\ntrain_aug = {self._config.train_aug}\neval_aug = {self._config.eval_aug}"
        # )
        # self._df = self._config.verify_and_get_df(schema=self.df_schema, fallback_df=ImagenetETL.load('index', 'hdf5', 'imagenette'))
        # self._split_df = self._config.verify_and_get_split_df(df=self._df, schema=self.split_df_schema, split=self._split)

    # def __len__(self) -> int:
        # return len(self._split_df)

    # def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        # idx_row = self._split_df.iloc[idx]
        # with h5py.File(self._root, mode="r") as hdf5_file:
            # image = iio.imread(BytesIO(hdf5_file["images"][idx_row["df_idx"]]))
        # image = np.stack((image,) * 3, axis=-1) if image.ndim == 2 else image
        # image = self._config.image_pre(image)
        # if self.split in ("train", "trainvaltest"):
            # image = self._config.train_aug(image)
        # elif self._split in ("val", "test"):
            # image = self._config.eval_aug(image)
        # return image, idx_row["label_idx"], idx_row["df_idx"]

    # @property
    # def root(self):
        # return self._root

    # @property
    # def split(self) -> Literal["train", "val", "test", "trainvaltest", "all"]:
        # return self._split

    # @property
    # def df(self) -> pd.DataFrame:
        # return self._df

    # @property
    # def split_df(self) -> pd.DataFrame:
        # return self._split_df

# class ImagenetteInMemoryClassification(Dataset):
    # name = "imagenette_inmemory_classification"
    # class_names = ImagenetETL.imagenette_class_names
    # num_classes = len(ImagenetETL.imagenette_class_names)
    # means = ImagenetETL.means
    # std_devs = ImagenetETL.std_devs
    # df_schema = pa.DataFrameSchema(
        # {
            # "image_path": pa.Column(str, coerce=True),
            # "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, len(ImagenetETL.imagenette_class_names))))),
            # "label_str": pa.Column(str),
            # "split": pa.Column(str, pa.Check.isin(Dataset.splits)),
        # },
        # index=pa.Index(int, unique=True),
    # )

    # split_df_schema = pa.DataFrameSchema(
        # {
            # "image_path": pa.Column(str, coerce=True),
            # "df_idx": pa.Column(int),
            # "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, len(ImagenetETL.imagenette_class_names))))),
        # },
        # index=pa.Index(int, unique=True),
    # )

    # def __init__(self, split: Literal["train", "val", "test", "trainvaltest", "all"] = "all", config: Optional[DatasetConfig] = None) -> None:
        # self._root = fs.get_valid_file_err(ImagenetETL.local, "hdf5", "imagenette.h5")
        # self._split = self.get_valid_split_err(split)
        # self._config = config or ImagenetETL.default_config
        # logger.info(
            # f"init {self.name}[{self._split}] using\nimage_pre = {self._config.image_pre}\ntarget_pre = {self._config.target_pre}\ntrain_aug = {self._config.train_aug}\neval_aug = {self._config.eval_aug}"
        # )
        # self._df = self._config.verify_and_get_df(schema=self.df_schema, fallback_df=ImagenetETL.load('index', 'hdf5', 'imagnette'))
        # self._split_df = self._config.verify_and_get_split_df(df=self._df, schema=self.split_df_schema, split=self._split)

        # with h5py.File(self._root, mode="r") as f:
            # self._images = f["images"][:]

    # def __len__(self):
        # return len(self._split_df)

    # def __getitem__(self, idx: int):
        # idx_row = self._split_df.iloc[idx]
        # image = iio.imread(BytesIO(self._images[idx_row["df_idx"]]))
        # image = np.stack((image,) * 3, axis=-1) if image.ndim == 2 else image
        # image = self._config.image_pre(image)
        # if self.split in ("train", "trainvaltest"):
            # image = self._config.train_aug(image)
        # elif self._split in ("val", "test"):
            # image = self._config.eval_aug(image)
        # return image, idx_row["label_idx"], idx_row["df_idx"]

    # @property
    # def root(self):
        # return self._root

    # @property
    # def split(self) -> Literal["train", "val", "test", "trainvaltest", "all"]:
        # return self._split

    # @property
    # def df(self) -> pd.DataFrame:
        # return self._df

    # @property
    # def split_df(self) -> pd.DataFrame:
        # return self._split_df