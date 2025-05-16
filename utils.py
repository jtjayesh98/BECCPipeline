from osgeo import gdal
import os
import rasterio
import ee
def get_image_dimension(image):
    dataset = gdal.Open(image)
    rows = dataset.RasterXSize
    cols = dataset.RasterYSize
    return rows, cols

def get_image_resolution(image):
    in_ds = gdal.Open(image)
    P = in_ds.GetGeoTransform()[1]
    return P


def print_tif_info(tif_path):
    """Prints metadata and other information about a TIFF file and displays it using folium."""
    with rasterio.open(tif_path) as src:
        print("File:", tif_path)
        print("CRS:", src.crs)
        print("Dimensions (Width x Height):", src.width, "x", src.height)
        print("Number of Bands:", src.count)
        print("Bounds:", src.bounds)
        print("Transform:", src.transform)

        for i in range(1, src.count + 1):
            band = src.read(i)
            print(f"\nBand {i} Statistics:")
            print("  Min:", band.min())
            print("  Max:", band.max())
            print("  Mean:", band.mean())
            print("  Data Type:", band.dtype)

class MapChecker:
    def __init__(self):
        self.image = None
        self.arr = None
        self.in_fn = None

    def get_image_resolution(self, image):
        in_ds = gdal.Open(image)
        P = in_ds.GetGeoTransform()[1]
        return P

    def get_image_dimensions(self, image):
        dataset = gdal.Open(image)
        cols = dataset.RasterXSize
        rows = dataset.RasterYSize
        return rows, cols

    def get_image_datatype(self, image):
        in_ds = gdal.Open(image)
        in_band = in_ds.GetRasterBand(1)
        datatype = gdal.GetDataTypeName(in_band.DataType)
        return datatype

    def get_image_max_min(self, image):
        in_ds = gdal.Open(image)
        in_band = in_ds.GetRasterBand(1)
        min, max= in_band.ComputeRasterMinMax()
        return min, max

    def find_unique_values(self, arr, limit=2):
        unique_values = set()
        for value in np.nditer(arr):
            unique_values.add(value.item())
            if len(unique_values) > limit:
                return False
        return True

    def check_binary_map(self, in_fn):
        '''
        Check if input image is binary map
        :param in_fn: input image
        :return: True if the file is a binary map, False otherwise
        '''
        file_extension = in_fn.split('.')[-1].lower()
        file_name, _ = os.path.splitext(in_fn)
        if file_extension == 'rst':
            with open(file_name + '.rdc', 'r') as read_file:
                rdc_content = read_file.read().lower()
                byte_or_integer_binary = "data type   : byte" in rdc_content or (
                        "data type   : integer" in rdc_content and "min. value  : 0" in rdc_content and "max. value  : 1" in rdc_content)
                float_binary = "data type   : real" in rdc_content and "min. value  : 0.0000000" in rdc_content and "max. value  : 1.0000000" in rdc_content
        elif file_extension == 'tif':
            datatype = self.get_image_datatype(in_fn)
            min_val, max_val = self.get_image_max_min(in_fn)
            byte_or_integer_binary = datatype in ['Byte', 'CInt16', 'CInt32', 'Int16', 'Int32', 'UInt16',
                                                      'UInt32'] and max_val == 1 and min_val == 0
            float_binary = datatype in ['Float32', 'Float64', 'CFloat32', 'CFloat64'] and max_val == 1.0000000 and min_val == 0.0000000

        if byte_or_integer_binary or (float_binary):
            # For float_binary, use find_unique_values function to check if data only have two unique values [0.0000000, 1.0000000].
            if float_binary:
                in_ds = gdal.Open(in_fn)
                in_band = in_ds.GetRasterBand(1)
                arr = in_band.ReadAsArray()
                # If more than two unique values are found, it's not a binary map, return False.
                if not self.find_unique_values(arr, 2):
                    return False
            # Binary map: byte_or_integer_binary or float_binary with two unique values [0.0000000, 1.0000000], it returns True.
            return True
        # For any other scenario, it returns False.
        return False


def maskS2clouds(image):
    """
    Function to mask clouds using the Sentinel-2 QA band
    Args:
        image: ee.Image, Sentinel-2 image
    Returns:
        ee.Image, cloud masked Sentinel-2 image
    """
    qa = image.select('QA60')
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    mask = qa.bitwiseAnd(cloudBitMask).eq(0) \
            .And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask).divide(10000)


def calculateNDVI(image):
    """
    Function to calculate NDVI for a Sentinel-2 image
    Args:
        image: ee.Image, Sentinel-2 image
    Returns:
        ee.Image, image with NDVI band added
    """
    nir = image.select('B8')
    red = image.select('B4')
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    return image.addBands(ndvi)


def applySNIC(image):
    """
    Function to apply SNIC clustering based on NDVI
    Args:
        image: ee.Image, Sentinel-2 image with NDVI band
    Returns:
        ee.Image, image with SNIC cluster band
    """
    ndvi = image.select('NDVI')
    snic = ee.Algorithms.Image.Segmentation.SNIC(
        image=ndvi,
        size=30,
        compactness=1,
        connectivity=8,
        neighborhoodSize=256
    )
    clusters = snic.select('clusters')
    return image.addBands(clusters.rename('SNIC_clusters'))


def applySNIC_with_meanNDVI(image):
    """
    Apply SNIC segmentation and compute mean NDVI per segment.
    Args:
        image: ee.Image with NDVI band
    Returns:
        ee.Image with SNIC mean NDVI band
    """
    ndvi = image.select('NDVI')

    # Apply SNIC
    snic = ee.Algorithms.Image.Segmentation.SNIC(
        image=ndvi,
        size=10,
        compactness=1,
        connectivity=8,
        neighborhoodSize=256
    )

    # Get cluster ID band
    clusters = snic.select('clusters').rename('SNIC_clusters')

    # Add the clusters to the image
    clustered_image = image.addBands(clusters)

    # Compute mean NDVI per SNIC cluster
    meanNDVI = ndvi.addBands(clusters).reduceConnectedComponents(
        reducer=ee.Reducer.mean(),
        labelBand='SNIC_clusters'
    ).rename('mean_NDVI_SNIC')

    # Add mean NDVI as a new band
    return image.addBands(meanNDVI)

def applyKMeans(image, num_clusters=7):
    """
    Function to apply K-means clustering on SNIC clusters to fix the number of strata
    Args:
        image: ee.Image, Sentinel-2 image with SNIC_clusters band
        num_clusters: int, number of desired strata
    Returns:
        ee.Image, image with K-means cluster band
    """
    # Select the SNIC clusters and NDVI for clustering
    # training_image = image.select(['SNIC_clusters', 'NDVI'])
    training_image = image.select([
        'mean_NDVI_SNIC'
    ])

    # Sample the image to create a FeatureCollection
    training_data = training_image.sample(
        region=geometry,
        scale=30,
        numPixels=5000,  # Adjust number of points as needed
        seed=0,
        geometries=True
    )

    # Create and train a K-means clusterer
    clusterer = ee.Clusterer.wekaKMeans(num_clusters).train(training_data)

    # Apply the clusterer to the image
    kmeans_clusters = image.cluster(clusterer).rename('KMeans_clusters')

    return image.addBands(kmeans_clusters)