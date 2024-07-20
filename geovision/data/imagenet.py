from typing import Literal, Optional

import torch
import zipfile
import shutil
import h5py  # type: ignore
import numpy as np
import pandas as pd
import pandera as pa
import imageio.v3 as iio
import torchvision.transforms.v2 as T  # type: ignore

from pathlib import Path

from tqdm import tqdm
from io import BytesIO
from .dataset import Dataset
from .config import DatasetConfig
from .utils import get_valid_split, get_df, get_split_df
from geovision.io.local import get_valid_file_err, get_valid_dir_err, get_new_dir

import logging
logger = logging.getLogger(__name__)

class Imagenet:
    local = Path.home() / "datasets" / "imagenet"
    archive = local / "archives" / "imagenet-object-localization-challenge.zip"
    # fmt: off
    class_names =  ("tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "cock",
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
    # fmt: on
    num_classes = len(class_names)
    means = (0.485, 0.456, 0.406)
    std_devs = (0.229, 0.224, 0.225)
    default_config = DatasetConfig(
        random_seed=42,
        test_sample=0,
        val_sample=0.05,
        tabular_sampling="imagefolder_notest",
        image_pre=T.Compose([
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Resize(224, antialias=True),
        ]),
        target_pre=T.Identity(),
        train_aug=T.RandomHorizontalFlip(0.5),
        eval_aug=T.Identity(),
    )
    erroneous_images = tuple()

    @classmethod
    def download(cls):
        r"""download and save imagenette2.tgz to :imagenette/archives/imagenet-object-localization-challenge.zip"""
        raise NotImplementedError

    @classmethod
    def transform_to_imagefolder(cls):
        """extracts files from archive to :imagenet/imagefolder, raises OSError if :imagenet/archives/imagenet-object-localization-challenge.zip not found"""

        imagefolder_path = get_new_dir(cls.local / "imagefolder")
        temp_path = get_new_dir(cls.local / "temp")

        # extract archive contents to temp
        with zipfile.ZipFile(get_valid_file_err(cls.archive)) as zf:
            zf.extractall(temp_path, [n for n in zf.namelist() if not n.endswith(".xml")])

        # move metadata files to imagefolder
        shutil.move(temp_path / "LOC_synset_mapping.txt", imagefolder_path / "LOC_synset_mapping.txt")
        shutil.move(temp_path / "LOC_val_solution.csv", imagefolder_path / "LOC_val_solution.csv")

        # prepare subdirs for imagefolder 
        temp_train = temp_path / "ILSVRC" / "Data" / "CLS-LOC" / "train"
        for class_synset in (c.stem for c in temp_train.iterdir()):
            get_new_dir(imagefolder_path, "train", class_synset)
            get_new_dir(imagefolder_path, "val", class_synset)
        test_dir = get_new_dir(imagefolder_path, "test")

        # move train
        for temp_path in tqdm(list(temp_train.rglob("*.JPEG")), desc=f"{imagefolder_path/"train"}"):
            image_path = imagefolder_path / "train" / temp_path.parent.name / f"{temp_path.stem}.jpg"
            shutil.move(temp_path, image_path)

        # move val
        val_df = (
            pd.read_csv(imagefolder_path / "LOC_val_solution.csv")
            .assign(temp_path=lambda df: df["ImageId"].apply(lambda x: temp_path / "ILSVRC" / "Data" / "CLS-LOC" / "val" / f"{x}.JPEG"))
            .assign(parent_dir=lambda df: df["PredictionString"].apply(lambda x: x.split(" ")[0]))
            .assign(image_path=lambda df: df.apply(lambda x: imagefolder_path / "val" / x["parent_dir"] / f"{x["ImageId"]}.jpg", axis=1))
        )
        for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc=f"{imagefolder_path/"val"}"):
            shutil.move(row["temp_path"], row["image_path"])

        # move test
        for temp_path in tqdm(list((temp_path / "ILSVRC" / "Data" / "CLS-LOC" / "test").rglob("*.JPEG")), desc=f"{imagefolder_path/"test"}"):
            shutil.move(temp_path, test_dir / f"{temp_path.stem}.jpg")

        # cleanup
        shutil.rmtree(temp_path)

    @classmethod
    def transform_to_hdf(cls, val_only: bool = False) -> None:
        """encodes files from archive to :root/hdf5/imagenet.h5, raises OSError if :imagenet/archives/imagenet-object-localization-challenge.zip not found"""
        hdf5 = get_new_dir(cls.local / "hdf5") / "imagenet.h5"
        df = cls.get_dataset_df_from_archive()
        df["image_path"] = df["image_path"].apply(lambda x: x.as_posix())
        if val_only:
            val_samples = df["image_path"].apply(lambda x: True if "val" in x else False)
            df = df[val_samples].assign(idx = lambda df: df.index).reset_index(drop = True)
            hdf5 = hdf5.parent / "imagenet_val.h5"
        df.to_hdf(hdf5, mode="w", key="df")

        with zipfile.ZipFile(get_valid_file_err(cls.archive)) as zf:
            with h5py.File(hdf5, mode="r+") as h5file:
                images = h5file.create_dataset("images", (len(df),), h5py.special_dtype(vlen=np.dtype("uint8")))
                for idx, row in tqdm(df.iterrows(), total=len(df)):
                    images[idx] = np.frombuffer(zf.read(row["image_path"]), dtype=np.uint8)

    @classmethod
    def get_dataset_df_from_archive(cls) -> pd.DataFrame:
        """generates and returns dataset_df from :imagenet/archive/imagenet-object-localization-challenge.zip, raises OSError if archive is not found"""
        with zipfile.ZipFile(get_valid_file_err(cls.archive)) as zf:
            train_df = (
                pd.DataFrame({"image_path": [Path(n) for n in zf.namelist() if n.endswith(".JPEG") and "train" in n]})
                .assign(label_synset = lambda df: df["image_path"].apply(lambda x: x.parent.stem))
            )
            val_df = (
                pd.read_csv(zf.open("LOC_val_solution.csv"))
                .assign(image_path = lambda df: df["ImageId"].apply(lambda x: Path("ILSVRC")/"Data"/"CLS-LOC"/"val"/f"{x}.JPEG"))
                .assign(label_synset = lambda df: df["PredictionString"].apply(lambda x: x.split(" ")[0]))
                .drop(columns = ["ImageId", "PredictionString"])
            )
            return pd.concat([train_df, val_df]).pipe(cls._get_dataset_df)

    @classmethod
    def get_dataset_df_from_imagefolder(cls) -> pd.DataFrame:
        """looks for/generates and returns dataset_df from :imagenet/imagefolder, raises OSError if imagefolder dir is not found/empty"""
        imagefolder = get_valid_dir_err(cls.local / "imagefolder")
        try:
            return pd.read_csv(get_valid_file_err(imagefolder / "labels.csv"))
        except OSError:
            df = (
                pd.DataFrame({"image_path": list((imagefolder / "train").rglob("*.jpg")) + list((imagefolder / "val").rglob("*.jpg"))})
                .assign(image_path=lambda df: df["image_path"].apply(lambda x: Path(x.parents[1].stem, x.parents[0].stem, x.name)))
                .assign(label_str=lambda df: df["image_path"].apply(lambda x: x.parent.stem))
                .pipe(cls._get_dataset_df)
            )
            df.to_csv(imagefolder / "labels.csv", index=False)
            return df

    @classmethod
    def _get_dataset_df(cls, df: pd.DataFrame):
        synsets = sorted(df["label_synset"].unique())
        return (
            df
            .sort_values("label_synset")
            .assign(label_idx=lambda df: df["label_synset"].apply(lambda x: synsets.index(x)))
            .assign(label_str=lambda df: df["label_idx"].apply(lambda x: cls.class_names[x]))
            .reset_index(drop=True)
        )

    @classmethod
    def get_dataset_df_from_hdf5(cls) -> pd.DataFrame:
        """returns imagenet dataset_df stored in :imagenet/hdf5/imagenet.h5/df, raises OSError if h5 file is not found, and KeyError if df is not found in h5"""
        return pd.read_hdf(get_valid_file_err(cls.local / "hdf5" / "imagenet.h5"), key="df", mode="r")  # type: ignore

    @classmethod
    def remove_erroneous_images(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(index=cls.erroneous_images).reset_index(drop=True)


class ImagenetImagefolderClassification(Dataset):
    name = "imagenet_imagefolder_classification"
    class_names = Imagenet.class_names
    num_classes = Imagenet.num_classes
    means = Imagenet.means
    std_devs = Imagenet.std_devs
    df_schema = pa.DataFrameSchema(
        {
            "image_path": pa.Column(str, coerce=True),
            "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, Imagenet.num_classes)))),
            "label_str": pa.Column(str, pa.Check.isin(Imagenet.class_names)),
            "split": pa.Column(str, pa.Check.isin(Dataset.valid_splits)),
        },
        index=pa.Index(int, unique=True),
    )

    split_df_schema = pa.DataFrameSchema(
        {
            "image_path": pa.Column(str, coerce=True),
            "df_idx": pa.Column(int, unique=True),
            "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, Imagenet.num_classes)))),
        },
        index=pa.Index(int, unique=True),
    )

    def __init__(
        self,
        split: Literal["train", "val", "trainval", "test", "all"] = "all",
        config: Optional[DatasetConfig] = None,
    ):
        self._root = get_valid_dir_err(Imagenet.local / "imagefolder")
        self._split = get_valid_split(split)
        self._config = config or Imagenet.default_config
        logger.info(
            f"init {self.name}[{self._split}]\nimage_pre = {self._config.image_pre}\ntarget_pre = {self._config.target_pre}\ntrain_aug = {self._config.train_aug}\neval_aug = {self._config.eval_aug}"
        )
        self._df = get_df(self._config, self.df_schema, Imagenet.get_dataset_df_from_imagefolder())
        self._df = Imagenet.remove_erroneous_images(self._df)
        self._split_df = get_split_df(self._df, self.split_df_schema, self._split, self._root)

    def __len__(self):
        return len(self._split_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        idx_row = self._split_df.iloc[idx]
        image = iio.imread(idx_row["image_path"]).squeeze()
        image = np.stack((image,) * 3, axis=-1) if image.ndim == 2 else image
        image = self._config.image_pre(image)
        if self._split == "train":
            image = self._config.train_aug(image)
        elif self._split in ("val", "test"):
            image = self._config.eval_aug(image)
        return image, idx_row["label_idx"], idx_row["df_idx"]

    @property
    def root(self):
        return self._root

    @property
    def split(self) -> Literal["train", "val", "trainval", "test", "all"]:
        return self._split

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def split_df(self) -> pd.DataFrame:
        return self._split_df


class ImagenetHDF5Classification(Dataset):
    name = "imagenet_hdf5_classification"
    class_names = Imagenet.class_names
    num_classes = Imagenet.num_classes
    means = Imagenet.means
    std_devs = Imagenet.std_devs
    df_schema = pa.DataFrameSchema(
        {
            "image_path": pa.Column(str, coerce=True),
            "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, Imagenet.num_classes)))),
            "label_str": pa.Column(str),
            "split": pa.Column(str, pa.Check.isin(Dataset.valid_splits)),
        },
        index=pa.Index(int, unique=True),
    )
    split_df_schema = pa.DataFrameSchema(
        {
            "image_path": pa.Column(str, coerce=True),
            "df_idx": pa.Column(int),
            "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, Imagenet.num_classes)))),
        },
        index=pa.Index(int, unique=True),
    )

    def __init__(
        self,
        split: Literal["train", "val", "trainval", "test", "all"] = "all",
        config: Optional[DatasetConfig] = None,
    ):
        self._root = get_valid_dir_err(Imagenet.local / "hdf5" / "imagenet.h5")
        self._split = get_valid_split(split)
        self._config = config or Imagenet.default_config
        logger.info(
            f"init {self.name}[{self._split}]\nimage_pre = {self._config.image_pre}\ntarget_pre = {self._config.target_pre}\ntrain_aug = {self._config.train_aug}\neval_aug = {self._config.eval_aug}"
        )
        self._df = get_df(self._config, self.df_schema, Imagenet.get_dataset_df_from_hdf5())
        self._df = Imagenet.remove_erroneous_images(self._df)
        self._split_df = get_split_df(self._df, self.split_df_schema, self._split, self._root)

    def __len__(self) -> int:
        return len(self._split_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        idx_row = self.split_df.iloc[idx]
        with h5py.File(self.root, mode="r") as hdf5_file:
            image = iio.imread(BytesIO(hdf5_file["images"][idx_row["df_idx"]]))
        image = np.stack((image,) * 3, axis=-1) if image.ndim == 2 else image
        image = self._config.image_pre(image)
        if self._split == "train":
            image = self._config.train_aug(image)
        elif self._split in ("val", "test"):
            image = self._config.eval_aug(image)
        return image, idx_row["label_idx"], idx_row["df_idx"]

    @property
    def root(self):
        return self._root

    @property
    def split(self) -> Literal["train", "val", "trainval", "test", "all"]:
        return self._split

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def split_df(self) -> pd.DataFrame:
        return self._split_df