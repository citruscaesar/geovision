class CDL:
    classes = {
        0: "Background", 1: "Corn", 2: "Cotton", 3: "Rice", 4: "Sorghum", 5: "Soybeans", 6: "Sunflower",
        10: "Peanuts", 11: "Tobacco", 12: "Sweet Corn", 13: "Pop or Orn Corn", 14: "Mint",
        21: "Barley", 22: "Durum Wheat", 23: "Spring Wheat", 24: "Winter Wheat", 25: "Other Small Grains",
        26: "Dbl Crop WinWht/Soybeans", 27: "Rye", 28: "Oats", 29: "Millet", 30: "Speltz",
        31: "Canola", 32: "Flaxseed", 33: "Safflower", 34: "Rape Seed", 35: "Mustard",
        36: "Alfalfa", 37: "Other Hay/Non Alfalfa", 38: "Camelina", 39: "Buckwheat",
        41: "Sugarbeets", 42: "Dry Beans", 43: "Potatoes", 44: "Other Crops", 45: "Sugarcane",
        46: "Sweet Potatoes", 47: "Misc Vegs & Fruits", 48: "Watermelons", 49: "Onions",
        50: "Cucumbers", 51: "Chick Peas", 52: "Lentils", 53: "Peas", 54: "Tomatoes",
        55: "Caneberries", 56: "Hops", 57: "Herbs", 58: "Clover/Wildflowers", 59: "Sod/Grass Seed",
        60: "Switchgrass", 61: "Fallow/Idle Cropland", 63: "Forest", 64: "Shrubland", 65: "Barren",
        66: "Cherries", 67: "Peaches", 68: "Apples", 69: "Grapes", 70: "Christmas Trees",
        71: "Other Tree Crops", 72: "Citrus", 74: "Pecans", 75: "Almonds", 76: "Walnuts", 77: "Pears",
        81: "Clouds/No Data", 82: "Developed", 83: "Water", 87: "Wetlands", 88: "Nonag/Undefined",
        92: "Aquaculture", 111: "Open Water", 112: "Perennial Ice/Snow",
        121: "Developed/Open Space", 122: "Developed/Low Intensity", 123: "Developed/Med Intensity",
        124: "Developed/High Intensity", 131: "Barren", 141: "Deciduous Forest", 142: "Evergreen Forest",
        143: "Mixed Forest", 152: "Shrubland", 176: "Grass/Pasture", 190: "Woody Wetlands",
        195: "Herbaceous Wetlands", 204: "Pistachios", 205: "Triticale", 206: "Carrots",
        207: "Asparagus", 208: "Garlic", 209: "Cantaloupes", 210: "Prunes", 211: "Olives",
        212: "Oranges", 213: "Honeydew Melons", 214: "Broccoli", 215: "Avocados", 216: "Peppers",
        217: "Pomegranates", 218: "Nectarines", 219: "Greens", 220: "Plums", 221: "Strawberries",
        222: "Squash", 223: "Apricots", 224: "Vetch", 225: "Dbl Crop WinWht/Corn",
        226: "Dbl Crop Oats/Corn", 227: "Lettuce", 228: "Dbl Crop Triticale/Corn", 229: "Pumpkins",
        230: "Dbl Crop Lettuce/Durum Wht", 231: "Dbl Crop Lettuce/Cantaloupe", 232: "Dbl Crop Lettuce/Cotton",
        233: "Dbl Crop Lettuce/Barley", 234: "Dbl Crop Durum Wht/Sorghum", 235: "Dbl Crop Barley/Sorghum",
        236: "Dbl Crop WinWht/Sorghum", 237: "Dbl Crop Barley/Corn", 238: "Dbl Crop WinWht/Cotton",
        239: "Dbl Crop Soybeans/Cotton", 240: "Dbl Crop Soybeans/Oats", 241: "Dbl Crop Corn/Soybeans",
        242: "Blueberries", 243: "Cabbage", 244: "Cauliflower", 245: "Celery", 246: "Radishes",
        247: "Turnips", 248: "Eggplants", 249: "Gourds", 250: "Cranberries", 254: "Dbl Crop Barley/Soybeans"
    }

    # Hex color codes from USDA NASS CDL visualization palette
    # Source: https://developers.google.com/earth-engine/datasets/catalog/USDA_NASS_CDL
    colors = {
        0: "#000000", 1: "#ffd400", 2: "#ff2626", 3: "#00a9e6", 4: "#ff9e0f", 5: "#267300", 6: "#ffff00",
        10: "#70a800", 11: "#00af4d", 12: "#e0a60f", 13: "#e0a60f", 14: "#80d4ff",
        21: "#e2007f", 22: "#8a6453", 23: "#d9b56c", 24: "#a87000", 25: "#d69dbc",
        26: "#737300", 27: "#ae017e", 28: "#a15889", 29: "#73004c", 30: "#d69dbc",
        31: "#d1ff00", 32: "#8099ff", 33: "#d6d600", 34: "#d1ff00", 35: "#00af4d",
        36: "#ffa5e2", 37: "#a5f28c", 38: "#00af4d", 39: "#d69dbc",
        41: "#a800e2", 42: "#a50000", 43: "#702600", 44: "#00af4d", 45: "#b27fff",
        46: "#702600", 47: "#ff6666", 48: "#ff6666", 49: "#ffcc66",
        50: "#ff6666", 51: "#00af4d", 52: "#00ddaf", 53: "#54ff00", 54: "#f2a377",
        55: "#ff6666", 56: "#00af4d", 57: "#7cd3ff", 58: "#e8bfff", 59: "#aeffdd",
        60: "#00af4d", 61: "#bfbf77", 63: "#93cc93", 64: "#c6d69e", 65: "#ccbfa3",
        66: "#ff00ff", 67: "#ff8eaa", 68: "#ba0000", 69: "#550070", 70: "#a80084",
        71: "#702600", 72: "#ff6666", 74: "#b27fff", 75: "#702600", 76: "#00a582", 77: "#b27fff",
        81: "#f2f2f2", 82: "#9c9c9c", 83: "#4d70a3", 87: "#00bfff", 88: "#93cc93",
        92: "#00ffff", 111: "#4d70a3", 112: "#d3e2f9",
        121: "#9c9c9c", 122: "#9c9c9c", 123: "#9c9c9c",
        124: "#9c9c9c", 131: "#ccbfa3", 141: "#93cc93", 142: "#93cc93",
        143: "#93cc93", 152: "#c6d69e", 176: "#e8ffbf", 190: "#7cd3ff",
        195: "#e8bfff", 204: "#702600", 205: "#d69dbc", 206: "#ff6666",
        207: "#ff6666", 208: "#ff6666", 209: "#ff6666", 210: "#702600", 211: "#702600",
        212: "#ff6666", 213: "#ff6666", 214: "#ff6666", 215: "#702600", 216: "#ff6666",
        217: "#702600", 218: "#702600", 219: "#ff6666", 220: "#702600", 221: "#ff6666",
        222: "#ff6666", 223: "#702600", 224: "#00af4d", 225: "#a87000",
        226: "#a87000", 227: "#ff6666", 228: "#a87000", 229: "#ff6666",
        230: "#a87000", 231: "#ff6666", 232: "#ff2626",
        233: "#a87000", 234: "#a87000", 235: "#a87000",
        236: "#a87000", 237: "#a87000", 238: "#a87000",
        239: "#267300", 240: "#267300", 241: "#ffd400",
        242: "#007777", 243: "#ff6666", 244: "#ff6666", 245: "#ff6666", 246: "#ff6666",
        247: "#ff6666", 248: "#ff6666", 249: "#ff6666", 250: "#ff6666", 254: "#267300"
    }


    @staticmethod
    def get_cdl_colors() -> dict:
        """
        Get the USDA NASS CDL color palette mapping class IDs to hex colors.

        The USDA solved the challenge of visualizing 133+ crop categories through a carefully designed color scheme based on
        perceptual grouping, semantic associations, and visual hierarchy.

        Color Scheme Patterns:

        1. Major Commodity Crops - Bright, Distinctive Colors:
           - Corn (1): #ffd400 - Bright golden yellow (highly visible)
           - Soybeans (5): #267300 - Dark green (foliage color)
           - Cotton (2): #ff2626 - Bright red (high contrast)
           - Rice (3): #00a9e6 - Cyan blue (flooded paddies)
           - Winter Wheat (24): #a87000 - Brown-tan (harvest color)

        2. Vegetables and Fruits - Red/Pink Family (#ff6666):
           Many vegetables share the same red-pink color for visual grouping:
           - Watermelons, Cucumbers, Lettuce, Broccoli, Peppers, etc.
           - Exception: Tomatoes (54): #f2a377 - Unique peachy-orange

        3. Tree Crops and Nuts - Dark Browns (#702600):
           Permanent orchards use earthy brown tones:
           - Almonds, Pecans, Prunes, Olives, Pistachios, Nectarines, Apricots

        4. Grains and Small Grains - Earth Tones:
           - Barley (21): #e2007f - Magenta
           - Wheat varieties: Browns and tans
           - Oats/Rye: Mauve and purple tones

        5. Double Cropping - Inherited from Primary Crop:
           Sequential crops use the color of the dominant/first crop:
           - "Dbl Crop WinWht/Corn" (225): #a87000 - wheat brown
           - "Dbl Crop Corn/Soybeans" (241): #ffd400 - corn yellow

        6. Natural Land Cover - Earth Greens and Blues:
           - Forest (63, 141-143): #93cc93 - Soft sage green
           - Shrubland: #c6d69e - Olive green
           - Grass/Pasture (176): #e8ffbf - Light yellow-green
           - Water (83, 111): #4d70a3 - Deep blue
           - Wetlands: Blues and purples

        7. Developed/Urban - Neutral Grays:
           All development levels (121-124): #9c9c9c - Medium gray

        8. Specialty Crops - Unique, Saturated Colors:
           High-value crops get memorable colors reflecting their fruit:
           - Grapes (69): #550070 - Deep purple
           - Cherries (66): #ff00ff - Bright magenta
           - Apples (68): #ba0000 - Deep red
           - Sugarbeets (41): #a800e2 - Vivid purple
           - Blueberries (242): #007777 - Teal

        9. Legumes and Forages - Green Family:
           - Alfalfa (36): #ffa5e2 - Pink (alfalfa flowers)
           - Peanuts (10): #70a800 - Yellow-green
           - Peas (53): #54ff00 - Bright lime green

        Design Philosophy:
        - Grouping minor crops reduces visual complexity
        - Semantic color coding (water=blue, developed=gray, forest=green)
        - Prioritizing major commodities with distinct, saturated colors
        - Using color families for related crops
        - Creating scannable, memorable, and hierarchical visualizations

        Returns:
            Dictionary mapping CDL class IDs to hex color codes
        """
        return CDL.colors


class NLCD:
    """
    National Land Cover Database (NLCD) - United States land cover classification.

    NLCD provides 30-meter resolution land cover data for the continental US. The dataset uses
    a modified Anderson Level II classification system with 16 land cover classes.

    Source: USGS National Land Cover Database
    GEE: USGS/NLCD_RELEASES/2021_REL/NLCD
    """

    classes = {
        11: "Open Water",
        12: "Perennial Ice/Snow",
        21: "Developed, Open Space",
        22: "Developed, Low Intensity",
        23: "Developed, Medium Intensity",
        24: "Developed, High Intensity",
        31: "Barren Land",
        41: "Deciduous Forest",
        42: "Evergreen Forest",
        43: "Mixed Forest",
        52: "Shrub/Scrub",
        71: "Grassland/Herbaceous",
        81: "Pasture/Hay",
        82: "Cultivated Crops",
        90: "Woody Wetlands",
        95: "Emergent Herbaceous Wetlands"
    }

    # Hex color codes from USGS NLCD visualization palette
    # Source: https://www.mrlc.gov and https://developers.google.com/earth-engine/datasets/catalog/USGS_NLCD_RELEASES_2021_REL_NLCD
    colors = {
        11: "#466b9f",  # Open Water - Deep blue
        12: "#d1def8",  # Perennial Ice/Snow - Light blue-white
        21: "#dec5c5",  # Developed, Open Space - Light pink-gray
        22: "#d99282",  # Developed, Low Intensity - Salmon pink
        23: "#eb0000",  # Developed, Medium Intensity - Bright red
        24: "#ab0000",  # Developed, High Intensity - Dark red
        31: "#b3ac9f",  # Barren Land - Gray-tan
        41: "#68ab5f",  # Deciduous Forest - Medium green
        42: "#1c5f2c",  # Evergreen Forest - Dark green
        43: "#b5c58f",  # Mixed Forest - Light olive green
        52: "#ccb879",  # Shrub/Scrub - Tan-gold
        71: "#dfdfc2",  # Grassland/Herbaceous - Light yellow-tan
        81: "#dcd939",  # Pasture/Hay - Yellow-green
        82: "#ab6c28",  # Cultivated Crops - Brown-orange
        90: "#b8d9eb",  # Woody Wetlands - Light cyan
        95: "#6c9fb8"   # Emergent Herbaceous Wetlands - Medium blue-gray
    }

    @staticmethod
    def get_nlcd_colors() -> dict:
        """
        Get the NLCD color palette mapping class IDs to hex colors.

        NLCD uses a simplified 16-class system designed for consistent land cover monitoring
        across the United States. The color scheme prioritizes:

        1. Water and Ice - Blue Tones:
           - Open Water (11): #466b9f - Deep blue
           - Perennial Ice/Snow (12): #d1def8 - Light icy blue
           - Wetlands (90, 95): Cyan and blue-gray tones

        2. Developed/Urban - Red Gradient by Intensity:
           - Open Space (21): #dec5c5 - Light pink-gray (parks, lawns)
           - Low Intensity (22): #d99282 - Salmon (suburban)
           - Medium Intensity (23): #eb0000 - Bright red (urban)
           - High Intensity (24): #ab0000 - Dark red (dense urban)

        3. Forests - Green Spectrum:
           - Deciduous (41): #68ab5f - Medium green (broadleaf)
           - Evergreen (42): #1c5f2c - Dark green (conifers)
           - Mixed (43): #b5c58f - Light olive (mixed stands)

        4. Vegetation/Agriculture - Yellow-Tan Tones:
           - Grassland (71): #dfdfc2 - Light tan
           - Pasture/Hay (81): #dcd939 - Yellow-green (managed)
           - Cultivated Crops (82): #ab6c28 - Brown-orange (row crops)
           - Shrub/Scrub (52): #ccb879 - Tan-gold

        5. Barren - Neutral Gray:
           - Barren Land (31): #b3ac9f - Gray-tan (rock, sand, bare ground)

        Design Philosophy:
        - Color intensity reflects land use intensity (light → dark for development)
        - Green shades distinguish forest types by dominant species
        - Yellow/brown tones for agricultural and grassland areas
        - Blue palette reserved for water features
        - Clear visual distinction between natural and developed areas

        Returns:
            Dictionary mapping NLCD class IDs to hex color codes
        """
        return NLCD.colors


class CORINE:
    """
    CORINE Land Cover - European land cover classification system.

    CORINE (Coordination of Information on the Environment) provides 100-meter resolution
    land cover data for Europe using a standardized 44-class nomenclature organized into
    5 hierarchical levels: artificial surfaces, agricultural areas, forest and semi-natural
    areas, wetlands, and water bodies.

    Source: European Environment Agency (EEA)
    GEE: COPERNICUS/CORINE/V20/100m
    """

    classes = {
        # 1. Artificial surfaces (1xx)
        111: "Continuous urban fabric",
        112: "Discontinuous urban fabric",
        121: "Industrial or commercial units",
        122: "Road and rail networks and associated land",
        123: "Port areas",
        124: "Airports",
        131: "Mineral extraction sites",
        132: "Dump sites",
        133: "Construction sites",
        141: "Green urban areas",
        142: "Sport and leisure facilities",

        # 2. Agricultural areas (2xx)
        211: "Non-irrigated arable land",
        212: "Permanently irrigated land",
        213: "Rice fields",
        221: "Vineyards",
        222: "Fruit trees and berry plantations",
        223: "Olive groves",
        231: "Pastures",
        241: "Annual crops associated with permanent crops",
        242: "Complex cultivation patterns",
        243: "Land principally occupied by agriculture with significant areas of natural vegetation",
        244: "Agro-forestry areas",

        # 3. Forest and semi-natural areas (3xx)
        311: "Broad-leaved forest",
        312: "Coniferous forest",
        313: "Mixed forest",
        321: "Natural grasslands",
        322: "Moors and heathland",
        323: "Sclerophyllous vegetation",
        324: "Transitional woodland-shrub",
        331: "Beaches, dunes, sands",
        332: "Bare rocks",
        333: "Sparsely vegetated areas",
        334: "Burnt areas",
        335: "Glaciers and perpetual snow",

        # 4. Wetlands (4xx)
        411: "Inland marshes",
        412: "Peat bogs",
        421: "Salt marshes",
        422: "Salines",
        423: "Intertidal flats",

        # 5. Water bodies (5xx)
        511: "Water courses",
        512: "Water bodies",
        521: "Coastal lagoons",
        522: "Estuaries",
        523: "Sea and ocean"
    }

    # Hex color codes from CORINE Land Cover visualization palette
    # Source: https://land.copernicus.eu and European Environment Agency
    colors = {
        # 1. Artificial surfaces - Reds, purples, and pinks
        111: "#e6004d",  # Continuous urban fabric - Magenta
        112: "#ff0000",  # Discontinuous urban fabric - Red
        121: "#cc4df2",  # Industrial or commercial units - Purple
        122: "#cc0000",  # Road and rail networks - Dark red
        123: "#e6cccc",  # Port areas - Light pink
        124: "#e6cce6",  # Airports - Light purple
        131: "#a600cc",  # Mineral extraction sites - Dark purple
        132: "#a64d00",  # Dump sites - Brown
        133: "#ff4dff",  # Construction sites - Bright pink
        141: "#a6f200",  # Green urban areas - Lime green
        142: "#a6ff80",  # Sport and leisure facilities - Light green

        # 2. Agricultural areas - Yellows and oranges
        211: "#ffffa8",  # Non-irrigated arable land - Pale yellow
        212: "#ffff00",  # Permanently irrigated land - Bright yellow
        213: "#e6e600",  # Rice fields - Yellow-green
        221: "#e68000",  # Vineyards - Orange
        222: "#f2a64d",  # Fruit trees and berry plantations - Light orange
        223: "#e6a600",  # Olive groves - Yellow-orange
        231: "#e6e64d",  # Pastures - Light yellow
        241: "#ffa6ff",  # Annual crops with permanent crops - Pink
        242: "#ffe6a6",  # Complex cultivation patterns - Cream
        243: "#e6cc4d",  # Agriculture with natural vegetation - Tan
        244: "#f2cca6",  # Agro-forestry areas - Beige

        # 3. Forest and semi-natural areas - Greens and earth tones
        311: "#80ff00",  # Broad-leaved forest - Bright green
        312: "#00a600",  # Coniferous forest - Dark green
        313: "#4dff00",  # Mixed forest - Medium green
        321: "#ccf24d",  # Natural grasslands - Yellow-green
        322: "#a6ffe6",  # Moors and heathland - Cyan-green
        323: "#a6e64d",  # Sclerophyllous vegetation - Olive
        324: "#a6f200",  # Transitional woodland-shrub - Lime
        331: "#e6e6e6",  # Beaches, dunes, sands - Light gray
        332: "#cccccc",  # Bare rocks - Gray
        333: "#ccffcc",  # Sparsely vegetated areas - Very light green
        334: "#000000",  # Burnt areas - Black
        335: "#a6e6cc",  # Glaciers and perpetual snow - Ice blue

        # 4. Wetlands - Blues and purples
        411: "#a6a6ff",  # Inland marshes - Light blue
        412: "#4d4dff",  # Peat bogs - Medium blue
        421: "#ccccff",  # Salt marshes - Pale blue
        422: "#e6e6ff",  # Salines - Very light blue
        423: "#a6a6e6",  # Intertidal flats - Blue-gray

        # 5. Water bodies - Blue spectrum
        511: "#00ccf2",  # Water courses - Cyan
        512: "#80f2e6",  # Water bodies - Light cyan
        521: "#00ffa6",  # Coastal lagoons - Turquoise
        522: "#a6ffe6",  # Estuaries - Light turquoise
        523: "#e6f2ff"   # Sea and ocean - Very pale blue
    }

    @staticmethod
    def get_corine_colors() -> dict:
        """
        Get the CORINE Land Cover color palette mapping class IDs to hex colors.

        CORINE uses a hierarchical 44-class system designed for pan-European land monitoring.
        The color scheme follows a systematic approach based on the 5 main categories:

        1. Artificial Surfaces (1xx) - Red/Purple/Pink Palette:
           Urban areas use red tones to maximize visibility:
           - Continuous urban fabric (111): #e6004d - Magenta (dense cities)
           - Discontinuous urban fabric (112): #ff0000 - Red (suburban)
           - Industrial units (121): #cc4df2 - Purple (commercial/industrial)
           - Transport infrastructure: Dark red tones
           - Green urban areas (141, 142): Lime/light greens (parks)

        2. Agricultural Areas (2xx) - Yellow/Orange Palette:
           Agricultural land uses warm tones reflecting crops and cultivation:
           - Arable land (211, 212): Yellow shades
           - Permanent crops (221-223): Orange tones (vineyards, orchards)
           - Pastures (231): Light yellow
           - Mixed agricultural (241-244): Cream and tan tones

        3. Forest and Semi-Natural Areas (3xx) - Green/Gray Palette:
           Natural vegetation and barren land:
           - Forests (311-313): Green spectrum by forest type
           - Grasslands and shrubs (321-324): Yellow-green to lime
           - Barren surfaces (331-333): Gray tones
           - Burnt areas (334): Black
           - Glaciers (335): Ice blue

        4. Wetlands (4xx) - Blue/Purple Palette:
           Wetland areas use soft blue and purple tones:
           - Inland wetlands (411, 412): Light to medium blue
           - Coastal wetlands (421-423): Pale blue and blue-gray

        5. Water Bodies (5xx) - Blue Spectrum:
           Water features use cyan to pale blue:
           - Rivers (511): Cyan
           - Lakes (512): Light cyan
           - Coastal waters (521-523): Turquoise to pale blue

        Design Philosophy:
        - Hierarchical color coding by main category (1xx, 2xx, 3xx, 4xx, 5xx)
        - Color intensity reflects land use intensity within each category
        - High contrast between categories for easy visual distinction
        - Semantic color associations (red=urban, yellow=agriculture, green=forest, blue=water)
        - Standardized across all European countries for consistent mapping

        Returns:
            Dictionary mapping CORINE class IDs to hex color codes
        """
        return CORINE.colors 