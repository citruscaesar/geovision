from typing import Literal, Optional 

import torch
import zipfile 
import shutil
import h5py # type: ignore
import numpy as np
import pandas as pd
import pandera as pa
import imageio.v3 as iio
import torchvision.transforms.v2 as T # type: ignore

from pathlib import Path

from tqdm import tqdm 
from io import BytesIO
from .dataset import Dataset, DatasetConfig, TransformsConfig, Validator
from geovision.io.local import get_valid_file_err, get_valid_dir_err, get_new_dir, is_valid_file
from geovision.logging import get_logger

logger = get_logger("imagenet")

class Imagenet:
    class_names = (
        'tench',  'goldfish',  'great_white_shark',  'tiger_shark',  'hammerhead', 
        'electric_ray',  'stingray',  'cock',  'hen',  'ostrich',  'brambling',  'goldfinch', 
        'house_finch',  'junco',  'indigo_bunting',  'robin',  'bulbul',  'jay',  'magpie', 
        'chickadee',  'water_ouzel',  'kite',  'bald_eagle',  'vulture',  'great_grey_owl', 
        'european_fire_salamander',  'common_newt',  'eft',  'spotted_salamander',  'axolotl',  'bullfrog', 
        'tree_frog',  'tailed_frog',  'loggerhead',  'leatherback_turtle',  'mud_turtle', 
        'terrapin',  'box_turtle',  'banded_gecko',  'common_iguana',  'american_chameleon', 
        'whiptail',  'agama',  'frilled_lizard',  'alligator_lizard',  'gila_monster', 
        'green_lizard',  'african_chameleon',  'komodo_dragon',  'african_crocodile', 
        'american_alligator',  'triceratops',  'thunder_snake',  'ringneck_snake',  'hognose_snake', 
        'green_snake',  'king_snake',  'garter_snake',  'water_snake',  'vine_snake',  'night_snake',
        'boa_constrictor',  'rock_python',  'indian_cobra',  'green_mamba', 
        'sea_snake',  'horned_viper',  'diamondback',  'sidewinder',  'trilobite',  'harvestman', 
        'scorpion',  'black_and_gold_garden_spider',  'barn_spider',  'garden_spider', 
        'black_widow',  'tarantula',  'wolf_spider',  'tick',  'centipede',  'black_grouse', 
        'ptarmigan',  'ruffed_grouse',  'prairie_chicken',  'peacock',  'quail',  'partridge', 
        'african_grey',  'macaw',  'sulphur-crested_cockatoo',  'lorikeet',  'coucal',  'bee_eater', 
        'hornbill',  'hummingbird',  'jacamar',  'toucan',  'drake',  'red-breasted_merganser', 
        'goose',  'black_swan',  'tusker',  'echidna',  'platypus',  'wallaby',  'koala', 
        'wombat',  'jellyfish',  'sea_anemone',  'brain_coral',  'flatworm',  'nematode', 
        'conch',  'snail',  'slug',  'sea_slug',  'chiton',  'chambered_nautilus', 
        'dungeness_crab',  'rock_crab',  'fiddler_crab',  'king_crab',  'american_lobster', 
        'spiny_lobster',  'crayfish',  'hermit_crab',  'isopod',  'white_stork',  'black_stork', 
        'spoonbill',  'flamingo',  'little_blue_heron',  'american_egret',  'bittern',  'crane', 
        'limpkin',  'european_gallinule',  'american_coot',  'bustard',  'ruddy_turnstone', 
        'red-backed_sandpiper',  'redshank',  'dowitcher',  'oystercatcher',  'pelican',  'king_penguin', 
        'albatross',  'grey_whale',  'killer_whale',  'dugong',  'sea_lion',  'chihuahua', 
        'japanese_spaniel',  'maltese_dog',  'pekinese',  'shih-tzu',  'blenheim_spaniel',  'papillon', 
        'toy_terrier',  'rhodesian_ridgeback',  'afghan_hound',  'basset',  'beagle',  'bloodhound', 
        'bluetick',  'black-and-tan_coonhound',  'walker_hound',  'english_foxhound',  'redbone', 
        'borzoi',  'irish_wolfhound',  'italian_greyhound',  'whippet',  'ibizan_hound', 
        'norwegian_elkhound',  'otterhound',  'saluki',  'scottish_deerhound',  'weimaraner', 
        'staffordshire_bullterrier',  'american_staffordshire_terrier',  'bedlington_terrier',  'border_terrier', 
        'kerry_blue_terrier',  'irish_terrier',  'norfolk_terrier',  'norwich_terrier', 
        'yorkshire_terrier',  'wire-haired_fox_terrier',  'lakeland_terrier',  'sealyham_terrier', 
        'airedale',  'cairn',  'australian_terrier',  'dandie_dinmont',  'boston_bull', 
        'miniature_schnauzer',  'giant_schnauzer',  'standard_schnauzer',  'scotch_terrier', 
        'tibetan_terrier',  'silky_terrier',  'soft-coated_wheaten_terrier', 
        'west_highland_white_terrier',  'lhasa',  'flat-coated_retriever',  'curly-coated_retriever', 
        'golden_retriever',  'labrador_retriever',  'chesapeake_bay_retriever', 
        'german_short-haired_pointer',  'vizsla',  'english_setter',  'irish_setter',  'gordon_setter', 
        'brittany_spaniel',  'clumber',  'english_springer',  'welsh_springer_spaniel',  'cocker_spaniel',
        'sussex_spaniel',  'irish_water_spaniel',  'kuvasz',  'schipperke', 
        'groenendael',  'malinois',  'briard',  'kelpie',  'komondor',  'old_english_sheepdog', 
        'shetland_sheepdog',  'collie',  'border_collie',  'bouvier_des_flandres',  'rottweiler', 
        'german_shepherd',  'doberman',  'miniature_pinscher',  'greater_swiss_mountain_dog', 
        'bernese_mountain_dog',  'appenzeller',  'entlebucher',  'boxer',  'bull_mastiff',  'tibetan_mastiff',
        'french_bulldog',  'great_dane',  'saint_bernard',  'eskimo_dog',  'malamute',
        'siberian_husky',  'dalmatian',  'affenpinscher',  'basenji',  'pug', 
        'leonberg',  'newfoundland',  'great_pyrenees',  'samoyed',  'pomeranian',  'chow', 
        'keeshond',  'brabancon_griffon',  'pembroke',  'cardigan',  'toy_poodle', 
        'miniature_poodle',  'standard_poodle',  'mexican_hairless',  'timber_wolf',  'white_wolf', 
        'red_wolf',  'coyote',  'dingo',  'dhole',  'african_hunting_dog',  'hyena',  'red_fox', 
        'kit_fox',  'arctic_fox',  'grey_fox',  'tabby',  'tiger_cat',  'persian_cat', 
        'siamese_cat',  'egyptian_cat',  'cougar',  'lynx',  'leopard',  'snow_leopard',  'jaguar', 
        'lion',  'tiger',  'cheetah',  'brown_bear',  'american_black_bear',  'ice_bear', 
        'sloth_bear',  'mongoose',  'meerkat',  'tiger_beetle',  'ladybug',  'ground_beetle', 
        'long-horned_beetle',  'leaf_beetle',  'dung_beetle',  'rhinoceros_beetle',  'weevil',  'fly', 
        'bee',  'ant',  'grasshopper',  'cricket',  'walking_stick',  'cockroach',  'mantis',
        'cicada',  'leafhopper',  'lacewing',  'dragonfly',  'damselfly',  'admiral', 
        'ringlet',  'monarch',  'cabbage_butterfly',  'sulphur_butterfly',  'lycaenid', 
        'starfish',  'sea_urchin',  'sea_cucumber',  'wood_rabbit',  'hare',  'angora', 
        'hamster',  'porcupine',  'fox_squirrel',  'marmot',  'beaver',  'guinea_pig',  'sorrel',
        'zebra',  'hog',  'wild_boar',  'warthog',  'hippopotamus',  'ox', 
        'water_buffalo',  'bison',  'ram',  'bighorn',  'ibex',  'hartebeest',  'impala',  'gazelle', 
        'arabian_camel',  'llama',  'weasel',  'mink',  'polecat',  'black-footed_ferret',  'otter', 
        'skunk',  'badger',  'armadillo',  'three-toed_sloth',  'orangutan',  'gorilla', 
        'chimpanzee',  'gibbon',  'siamang',  'guenon',  'patas',  'baboon',  'macaque',  'langur', 
        'colobus',  'proboscis_monkey',  'marmoset',  'capuchin',  'howler_monkey',  'titi', 
        'spider_monkey',  'squirrel_monkey',  'madagascar_cat',  'indri',  'indian_elephant', 
        'african_elephant',  'lesser_panda',  'giant_panda',  'barracouta',  'eel',  'coho', 
        'rock_beauty',  'anemone_fish',  'sturgeon',  'gar',  'lionfish',  'puffer',  'abacus', 
        'abaya',  'academic_gown',  'accordion',  'acoustic_guitar',  'aircraft_carrier', 
        'airliner',  'airship',  'altar',  'ambulance',  'amphibian',  'analog_clock',  'apiary', 
        'apron',  'ashcan',  'assault_rifle',  'backpack',  'bakery',  'balance_beam', 
        'balloon',  'ballpoint',  'band_aid',  'banjo',  'bannister',  'barbell', 
        'barber_chair',  'barbershop',  'barn',  'barometer',  'barrel',  'barrow',  'baseball', 
        'basketball',  'bassinet',  'bassoon',  'bathing_cap',  'bath_towel',  'bathtub', 
        'beach_wagon',  'beacon',  'beaker',  'bearskin',  'beer_bottle',  'beer_glass', 
        'bell_cote',  'bib',  'bicycle-built-for-two',  'bikini',  'binder',  'binoculars', 
        'birdhouse',  'boathouse',  'bobsled',  'bolo_tie',  'bonnet',  'bookcase',  'bookshop', 
        'bottlecap',  'bow',  'bow_tie',  'brass',  'brassiere',  'breakwater',  'breastplate', 
        'broom',  'bucket',  'buckle',  'bulletproof_vest',  'bullet_train',  'butcher_shop', 
        'cab',  'caldron',  'candle',  'cannon',  'canoe',  'can_opener',  'cardigan', 
        'car_mirror',  'carousel',  "carpenter's_kit",  'carton',  'car_wheel',  'cash_machine', 
        'cassette',  'cassette_player',  'castle',  'catamaran',  'cd_player',  'cello', 
        'cellular_telephone',  'chain',  'chainlink_fence',  'chain_mail',  'chain_saw',  'chest', 
        'chiffonier',  'chime',  'china_cabinet',  'christmas_stocking',  'church',  'cinema', 
        'cleaver',  'cliff_dwelling',  'cloak',  'clog',  'cocktail_shaker',  'coffee_mug', 
        'coffeepot',  'coil',  'combination_lock',  'computer_keyboard',  'confectionery', 
        'container_ship',  'convertible',  'corkscrew',  'cornet',  'cowboy_boot',  'clistowboy_hat', 
        'cradle',  'crane',  'crash_helmet',  'crate',  'crib',  'crock_pot',  'croquet_ball', 
        'crutch',  'cuirass',  'dam',  'desk',  'desktop_computer',  'dial_telephone', 
        'diaper',  'digital_clock',  'digital_watch',  'dining_table',  'dishrag', 
        'dishwasher',  'disk_brake',  'dock',  'dogsled',  'dome',  'doormat',  'drilling_platform',
        'drum',  'drumstick',  'dumbbell',  'dutch_oven',  'electric_fan', 
        'electric_guitar',  'electric_locomotive',  'entertainment_center',  'envelope', 
        'espresso_maker',  'face_powder',  'feather_boa',  'file',  'fireboat',  'fire_engine', 
        'fire_screen',  'flagpole',  'flute',  'folding_chair',  'football_helmet',  'forklift', 
        'fountain',  'fountain_pen',  'four-poster',  'freight_car',  'french_horn', 
        'frying_pan',  'fur_coat',  'garbage_truck',  'gasmask',  'gas_pump',  'goblet',  'go-kart',
        'golf_ball',  'golfcart',  'gondola',  'gong',  'gown',  'grand_piano', 
        'greenhouse',  'grille',  'grocery_store',  'guillotine',  'hair_slide',  'hair_spray', 
        'half_track',  'hammer',  'hamper',  'hand_blower',  'hand-held_computer',  'handkerchief', 
        'hard_disc',  'harmonica',  'harp',  'harvester',  'hatchet',  'holster',  'home_theater', 
        'honeycomb',  'hook',  'hoopskirt',  'horizontal_bar',  'horse_cart',  'hourglass', 
        'ipod',  'iron',  "jack-o'-lantern",  'jean',  'jeep',  'jersey',  'jigsaw_puzzle', 
        'jinrikisha',  'joystick',  'kimono',  'knee_pad',  'knot',  'lab_coat',  'ladle', 
        'lampshade',  'laptop',  'lawn_mower',  'lens_cap',  'letter_opener',  'library', 
        'lifeboat',  'lighter',  'limousine',  'liner',  'lipstick',  'loafer',  'lotion', 
        'loudspeaker',  'loupe',  'lumbermill',  'magnetic_compass',  'mailbag',  'mailbox', 
        'maillot',  'maillot',  'manhole_cover',  'maraca',  'marimba',  'mask',  'matchstick', 
        'maypole',  'maze',  'measuring_cup',  'medicine_chest',  'megalith',  'microphone', 
        'microwave',  'military_uniform',  'milk_can',  'minibus',  'miniskirt',  'minivan', 
        'missile',  'mitten',  'mixing_bowl',  'mobile_home',  'model_t',  'modem',  'monastery',
        'monitor',  'moped',  'mortar',  'mortarboard',  'mosque',  'mosquito_net', 
        'motor_scooter',  'mountain_bike',  'mountain_tent',  'mouse',  'mousetrap',  'moving_van', 
        'muzzle',  'nail',  'neck_brace',  'necklace',  'nipple',  'notebook',  'obelisk', 
        'oboe',  'ocarina',  'odometer',  'oil_filter',  'organ',  'oscilloscope', 
        'overskirt',  'oxcart',  'oxygen_mask',  'packet',  'paddle',  'paddlewheel',  'padlock', 
        'paintbrush',  'pajama',  'palace',  'panpipe',  'paper_towel',  'parachute', 
        'parallel_bars',  'park_bench',  'parking_meter',  'passenger_car',  'patio',  'pay-phone', 
        'pedestal',  'pencil_box',  'pencil_sharpener',  'perfume',  'petri_dish',  'photocopier',
        'pick',  'pickelhaube',  'picket_fence',  'pickup',  'pier',  'piggy_bank', 
        'pill_bottle',  'pillow',  'ping-pong_ball',  'pinwheel',  'pirate',  'pitcher',  'plane', 
        'planetarium',  'plastic_bag',  'plate_rack',  'plow',  'plunger',  'polaroid_camera', 
        'pole',  'police_van',  'poncho',  'pool_table',  'pop_bottle',  'pot', 
        "potter's_wheel",  'power_drill',  'prayer_rug',  'printer',  'prison',  'projectile', 
        'projector',  'puck',  'punching_bag',  'purse',  'quill',  'quilt',  'racer',  'racket', 
        'radiator',  'radio',  'radio_telescope',  'rain_barrel',  'recreational_vehicle', 
        'reel',  'reflex_camera',  'refrigerator',  'remote_control',  'restaurant', 
        'revolver',  'rifle',  'rocking_chair',  'rotisserie',  'rubber_eraser',  'rugby_ball', 
        'rule',  'running_shoe',  'safe',  'safety_pin',  'saltshaker',  'sandal',  'sarong', 
        'sax',  'scabbard',  'scale',  'school_bus',  'schooner',  'scoreboard',  'screen', 
        'screw',  'screwdriver',  'seat_belt',  'sewing_machine',  'shield',  'shoe_shop', 
        'shoji',  'shopping_basket',  'shopping_cart',  'shovel',  'shower_cap', 
        'shower_curtain',  'ski',  'ski_mask',  'sleeping_bag',  'slide_rule',  'sliding_door',  'slot',
        'snorkel',  'snowmobile',  'snowplow',  'soap_dispenser',  'soccer_ball', 
        'sock',  'solar_dish',  'sombrero',  'soup_bowl',  'space_bar',  'space_heater', 
        'space_shuttle',  'spatula',  'speedboat',  'spider_web',  'spindle',  'sports_car', 
        'spotlight',  'stage',  'steam_locomotive',  'steel_arch_bridge',  'steel_drum', 
        'stethoscope',  'stole',  'stone_wall',  'stopwatch',  'stove',  'strainer',  'streetcar', 
        'stretcher',  'studio_couch',  'stupa',  'submarine',  'suit',  'sundial',  'sunglass', 
        'sunglasses',  'sunscreen',  'suspension_bridge',  'swab',  'sweatshirt', 
        'swimming_trunks',  'swing',  'switch',  'syringe',  'table_lamp',  'tank',  'tape_player', 
        'teapot',  'teddy',  'television',  'tennis_ball',  'thatch',  'theater_curtain', 
        'thimble',  'thresher',  'throne',  'tile_roof',  'toaster',  'tobacco_shop', 
        'toilet_seat',  'torch',  'totem_pole',  'tow_truck',  'toyshop',  'tractor', 
        'trailer_truck',  'tray',  'trench_coat',  'tricycle',  'trimaran',  'tripod', 
        'triumphal_arch',  'trolleybus',  'trombone',  'tub',  'turnstile',  'typewriter_keyboard', 
        'umbrella',  'unicycle',  'upright',  'vacuum',  'vase',  'vault',  'velvet', 
        'vending_machine',  'vestment',  'viaduct',  'violin',  'volleyball',  'waffle_iron', 
        'wall_clock',  'wallet',  'wardrobe',  'warplane',  'washbasin',  'washer',  'water_bottle',
        'water_jug',  'water_tower',  'whiskey_jug',  'whistle',  'wig', 
        'window_screen',  'window_shade',  'windsor_tie',  'wine_bottle',  'wing',  'wok', 
        'wooden_spoon',  'wool',  'worm_fence',  'wreck',  'yawl',  'yurt',  'web_site', 
        'comic_book',  'crossword_puzzle',  'street_sign',  'traffic_light',  'book_jacket', 
        'menu',  'plate',  'guacamole',  'consomme',  'hot_pot',  'trifle',  'ice_cream', 
        'ice_lolly',  'french_loaf',  'bagel',  'pretzel',  'cheeseburger',  'hotdog', 
        'mashed_potato',  'head_cabbage',  'broccoli',  'cauliflower',  'zucchini', 
        'spaghetti_squash',  'acorn_squash',  'butternut_squash',  'cucumber',  'artichoke', 
        'bell_pepper',  'cardoon',  'mushroom',  'granny_smith',  'strawberry',  'orange',  'lemon', 
        'fig',  'pineapple',  'banana',  'jackfruit',  'custard_apple',  'pomegranate', 
        'hay',  'carbonara',  'chocolate_sauce',  'dough',  'meat_loaf',  'pizza',  'potpie',
        'burrito',  'red_wine',  'espresso',  'cup',  'eggnog',  'alp',  'bubble', 
        'cliff',  'coral_reef',  'geyser',  'lakeside',  'promontory',  'sandbar',  'seashore',
        'valley',  'volcano',  'ballplayer',  'groom',  'scuba_diver',  'rapeseed', 
        'daisy',  "yellow_lady's_slipper",  'corn',  'acorn',  'hip',  'buckeye', 
        'coral_fungus',  'agaric',  'gyromitra',  'stinkhorn',  'earthstar',  'hen-of-the-woods', 
        'bolete',  'ear',  'toilet_tissue'
    ) 

    means = (0.485, 0.456, 0.406)
    std_devs = (0.229, 0.224, 0.225)
    num_classes = len(class_names)
    default_config = DatasetConfig(
        random_seed = 42,
        test_sample = 0.2,
        val_sample = 0.1,
        tabular_sampling = "stratified"
    )
    default_transforms  = TransformsConfig(
        image_transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale = True)]),
        common_transform = T.Resize((224, 224), antialias=True)
    )

    @classmethod
    def download(cls, root: str | Path):
        pass

    @classmethod
    def transform_to_imagefolder(cls, root: str | Path) -> None:
        imagefolder = get_new_dir(root, "imagefolder")
        temp_dir = get_new_dir(root, "temp")

        # NOTE: do not do i/o bound stuff on WSL, it's slow as shit
        # NOTE: extraction is riduculously slow, only extract the train+val dirs and the .csv
        # NOTE: consider direct extraction to imagefolder if the ZipFile.namelist() works and dosen't crash jupyter

        archive = get_valid_file_err(root, "archives", "imagenet-object-localization-challenge.zip", valid_extns=(".zip",))
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(temp_dir)

        shutil.move(temp_dir / "LOC_synset_mapping.txt", imagefolder / "LOC_synset_mapping.txt" )
        shutil.move(temp_dir / "LOC_val_solution.csv", imagefolder / "LOC_val_solution.csv" )
        
        temp_train_dir = temp_dir/"ILSVRC"/"Data"/"CLS-LOC"/"train"
        for temp_class_dir in temp_train_dir.iterdir():
            get_new_dir(imagefolder, "images", temp_class_dir.stem)

        temp_train_paths = list(temp_train_dir.rglob("*.JPEG"))
        for temp_path in tqdm(temp_train_paths, desc = "moving train"):
            image_path = imagefolder/"images"/temp_path.parent.name/ f"{temp_path.stem}.jpg" 
            shutil.move(temp_path, image_path)

        val_df = pd.read_csv(temp_dir / "LOC_val_solution.csv")
        for _, row in tqdm(val_df.iterrows(), total = len(val_df), desc = "moving val"):
            temp_path = temp_dir/"ILSVRC"/"Data"/"CLS-LOC"/"val"/f"{row["ImageId"]}.JPEG"
            parent_dir = row["PredictionString"].split(' ')[0]
            image_path = imagefolder/"images"/parent_dir/f"{parent_dir}_{temp_path.stem.split('_')[-1]}.jpg"
            shutil.move(temp_path, image_path)
        
        temp_test_paths = list((temp_dir/"ILSVRC"/"Data"/"CLS-LOC"/"test").rglob("*.JPEG"))
        test_dir = get_new_dir(imagefolder, "test")
        for temp_path in tqdm(temp_test_paths, desc = "moving test"):
            image_path = test_dir/f"{temp_path.stem}.jpg"
            shutil.move(temp_path, image_path)
        shutil.rmtree(temp_dir)

    @classmethod
    def transform_to_hdf(cls, root: str | Path) -> None:
        imagefolder_path = get_valid_dir_err(root, "imagefolder")
        hdf5_path = get_new_dir(root, "hdf5") / "imagenet.h5"

        df = cls.get_dataset_df_from_imagefolder(root)  
        df.to_hdf(hdf5_path, mode = "w", key = "df")
        with h5py.File(hdf5_path, mode = "r+") as hdf5_file:
            images = hdf5_file.create_dataset(
                name = "images", 
                shape = (1331167,),
                dtype = h5py.special_dtype(vlen = np.dtype('uint8'))
            )
            for idx, row in tqdm(df.iterrows(), total = len(df)):
                with open(imagefolder_path/row["image_path"], "rb") as image_file:
                    image_bytes = image_file.read()
                    images[idx] = np.frombuffer(image_bytes, dtype = np.uint8)

    @classmethod
    def get_dataset_df_from_archive(cls, root: str | Path) -> pd.DataFrame:
        archive = get_valid_file_err(root, "archives", "imagenet-object-localization-challenge.zip", valid_extns=(".zip",))
        raise NotImplementedError("cant figure out how to open imagenet zipfile without kernel crashing")

    @classmethod
    def get_dataset_df_from_imagefolder(cls, root: str | Path) -> pd.DataFrame:
        imagefolder = get_valid_dir_err(root, "imagefolder", "images")
        try:
            return pd.read_csv(get_valid_file_err(imagefolder.parent, "labels.csv", valid_extns=(".csv",)))
        except OSError:
            class_synsets = sorted(x.stem for x in imagefolder.iterdir())
            class_labels = {k:v for k, v in zip(class_synsets, cls.class_names)}
            image_paths = list(imagefolder.rglob("*.jpg"))
            df = (
                pd.DataFrame({"image_path": image_paths})
                .assign(image_path = lambda df: df["image_path"].apply(lambda x: Path(x.parents[1].stem, x.parents[0].stem, x.name)))
                .assign(label_str = lambda df: df["image_path"].apply(lambda x: x.parent.stem))
                .sort_values("label_str")
                .assign(label_idx = lambda df: df["label_str"].apply(lambda x: class_synsets.index(x)))
                .assign(label_str = lambda df: df["label_str"].apply(lambda x: class_labels[x]))
                .assign(split_on = lambda df: df["label_str"])
                .reset_index(drop = True)
            )
            df.to_csv(imagefolder.parent/"labels.csv", index = False)
            return df

    @classmethod
    def get_dataset_df_from_hdf5(cls, root: str | Path) -> pd.DataFrame:
        hdf5_path = get_valid_file_err(root / "hdf5" / "imagenet.h5", valid_extns=(".h5", ".hdf5"))
        return pd.read_hdf(hdf5_path, key = "df", mode = 'r') # type: ignore

class ImagenetImagefolderClassification(Dataset):
    name = "imagenet_imagefolder_classification"
    class_names = Imagenet.class_names
    num_classes = Imagenet.num_classes
    means = Imagenet.means 
    std_devs = Imagenet.std_devs 

    df_schema = pa.DataFrameSchema({
        "image_path": pa.Column(str, coerce = True),  
        "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, Imagenet.num_classes)))),
        "label_str": pa.Column(str, pa.Check.isin(Imagenet.class_names)),
        "split": pa.Column(str, pa.Check.isin(Dataset.valid_splits))
    }, index = pa.Index(int, unique=True))

    split_df_schema = pa.DataFrameSchema({
        "image_path": pa.Column(str, coerce = True),        
        "df_idx": pa.Column(int, unique = True),
        "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, Imagenet.num_classes)))),
    }, index = pa.Index(int, unique=True))

    def __init__(
            self,
            root: Path,
            split: Literal["train", "val", "trainval", "test", "all"] = "all",
            df: Optional[pd.DataFrame] = None,
            config: Optional[DatasetConfig] = None,
            transforms: Optional[TransformsConfig] = None,
    ) -> None:
        logger.debug(f"init {self.name}")
        self._root = Validator._get_root_dir(root/"imagefolder")
        self._split = Validator._get_split(split)
        self._transforms = Validator._get_transforms(transforms, Imagenet.default_transforms)
        self._df = Validator._get_df(
            df = df,
            config = config,
            schema = self.df_schema,
            default_df = Imagenet.get_dataset_df_from_imagefolder(root),
            default_config = Imagenet.default_config,
        )
        self._split_df = Validator._get_imagefolder_split_df(
            df = self._df,
            schema = self.split_df_schema,
            root = self._root,
            split = self._split
        )
    
    def __len__(self) :
        return len(self._split_df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        idx_row = self._split_df.iloc[idx]
        image = iio.imread(idx_row["image_path"]).squeeze()
        image = np.stack((image,)*3, axis = -1) if image.ndim == 2 else image
        image = self._transforms.image_transform(image) # type: ignore
        if self._split == "train" and self._transforms.common_transform is not None:
            image = self._transforms.common_transform(image)
        return image, idx_row["label_idx"], idx_row["df_idx"] # type: ignore 

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

    @property
    def transforms(self) -> TransformsConfig:
        return self._transforms

class ImagenetHDF5Classification(Dataset):
    name = "imagenet_hdf5_classification" 
    class_names = Imagenet.class_names
    num_classes = Imagenet.num_classes 
    means = Imagenet.means
    std_devs = Imagenet.std_devs
    df_schema = pa.DataFrameSchema({
        "image_path": pa.Column(str, coerce = True),
        "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, Imagenet.num_classes)))),
        "label_str": pa.Column(str),
        "split": pa.Column(str, pa.Check.isin(Dataset.valid_splits))
    }, index = pa.Index(int, unique=True))
    split_df_schema = pa.DataFrameSchema({
        "image_path": pa.Column(str, coerce = True),
        "df_idx": pa.Column(int),
        "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, Imagenet.num_classes)))),
    }, index = pa.Index(int, unique=True))

    def __init__(
            self, 
            root: Path,
            df: Optional[pd.DataFrame] = None,
            split: Literal["train", "val", "trainval", "test", "all"] = "all",
            config: Optional[DatasetConfig] = None,
            transforms: Optional[TransformsConfig] = None,
            **kwargs
        ) -> None:
        logger.debug(f"init {self.name}")
        self._root = Validator._get_root_hdf5(root/"hdf5"/"imagenet.h5")
        self._split = Validator._get_split(split)
        self._transforms = Validator._get_transforms(transforms, Imagenet.default_transforms)
        self._df = Validator._get_df(
            df = df,
            config = config,
            schema = self.df_schema,
            default_df = Imagenet.get_dataset_df_from_hdf5(root),
            default_config = Imagenet.default_config,
        )
        self._split_df = Validator._get_imagefolder_split_df(
            df = self._df,
            schema = self.split_df_schema,
            root = self._root,
            split = self._split
        )
        
    def __len__(self) -> int:
        return len(self._split_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        idx_row = self.split_df.iloc[idx]
        with h5py.File(self.root, mode = "r") as hdf5_file:
            image = iio.imread(BytesIO(hdf5_file["images"][idx_row["df_idx"]]))
        image = np.stack((image,)*3, axis = -1) if image.ndim == 2 else image
        image = self._transforms.image_transform(image) # type: ignore
        if self.split == "train" and self._transforms.common_transform is not None:
            image = self._transforms.common_transform(image)
        return image, idx_row["label_idx"], idx_row["df_idx"] # type: ignore

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

    @property
    def transforms(self) -> TransformsConfig:
        return self._transforms