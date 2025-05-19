from osgeo import gdal
import os
import rasterio
import ee

import zipfile
import os
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
from collections import defaultdict
import geopandas as gpd
from lxml import etree
from shapely.geometry import Point, LineString, Polygon
import pandas as pd
import tempfile


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









# Input paths

def extract_kml_from_kmz(kmz_path, output_dir='temp_kml'):
    """Extract KML file from KMZ archive."""
    if not os.path.exists(kmz_path):
        raise FileNotFoundError(f"KMZ file not found: {kmz_path}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        with zipfile.ZipFile(kmz_path, 'r') as kmz:
            kml_file = [f for f in kmz.namelist() if f.endswith('.kml')][0]
            kmz.extract(kml_file, output_dir)
            return os.path.join(output_dir, kml_file)
    except Exception as e:
        print(f"Error extracting KMZ: {e}")
        return None

def parse_site_geometry(kml_path, site_name):
    """Parse KML file and extract geometry for Adhapalli."""
    namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}
    try:
        tree = etree.parse(kml_path)
        root = tree.getroot()
        placemarks = root.findall('.//kml:Placemark', namespaces)

        for placemark in placemarks:
            name = placemark.find('kml:name', namespaces)
            if name is not None and site_name in name.text:
                for geom_type in ['Point', 'LineString', 'Polygon']:
                    geometry = placemark.find(f'.//kml:{geom_type}', namespaces)
                    if geometry is not None:
                        coords = geometry.find('kml:coordinates', namespaces)
                        if coords is not None:
                            coord_list = []
                            for coord in coords.text.strip().split():
                                lon, lat, *alt = map(float, coord.split(','))
                                coord_list.append((lon, lat))

                            # Create geometry based on type
                            if geom_type == 'Point':
                                return Point(coord_list[0])
                            elif geom_type == 'LineString':
                                return LineString(coord_list)
                            else:  # Polygon
                                return Polygon(coord_list)
        print("No " + site_name + " geometry found in KML")
        return None
    except Exception as e:
        print(f"Error parsing KML: {e}")
        return None

def reproject_raster(src_path, dst_path, dst_crs):
    """Reproject a raster to a specified CRS."""
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

def match_resolution_and_extent(def_path, cluster_shape, cluster_transform, cluster_crs):
    """Resample the deforestation raster to match the cluster raster's shape, transform, and CRS."""
    with rasterio.open(def_path) as def_src:
        def_data = np.empty(cluster_shape, dtype=def_src.meta['dtype'])

        reproject(
            source=rasterio.band(def_src, 1),
            destination=def_data,
            src_transform=def_src.transform,
            src_crs=def_src.crs,
            dst_transform=cluster_transform,
            dst_crs=cluster_crs,
            resampling=Resampling.nearest
        )
        return def_data, def_src.nodata

def calculate_deforestation_by_cluster(clusters_tif_path, deforestation_tif_path, kmz_path, output_pth):
    """Calculate deforestation within each cluster and save to CSV."""
    # Initialize temp_defor_path to avoid undefined variable errors
    temp_defor_path = None

    try:
        # Validate input files
        # print(deforestation_tif_path)
        for path in [clusters_tif_path, deforestation_tif_path, kmz_path]:
            if not os.path.exists(path):
                # raise FileNotFoundError(f"File not found: {path}")
                print(path)
        # Extract KML and get Adhapalli geometry
        kml_path = extract_kml_from_kmz(kmz_path)
        if not kml_path:
            return

        geometry = parse_site_geometry(kml_path)
        if geometry is None:
            return

        # Clean up temporary KML
        try:
            os.remove(kml_path)
            os.rmdir(os.path.dirname(kml_path))
        except:
            pass

        # Create GeoDataFrame with the geometry
        gdf = gpd.GeoDataFrame(geometry=[geometry], crs='EPSG:4326')

        # Load the clusters raster
        with rasterio.open(clusters_tif_path) as cluster_src:
            cluster_crs = cluster_src.crs
            cluster_nodata = cluster_src.nodata if cluster_src.nodata is not None else -9999
            cluster_bounds = cluster_src.bounds
            cluster_shape = cluster_src.shape
            cluster_transform = cluster_src.transform
            cluster_pixel_size = (cluster_transform[0], -cluster_transform[4])
            print("Cluster pixel size:", cluster_pixel_size)

        # Reproject geometry to cluster CRS
        if cluster_crs and cluster_crs != 'EPSG:4326':
            gdf = gdf.to_crs(cluster_crs)

        # Check if geometry intersects with cluster raster bounds
        geom_bounds = gdf.geometry.iloc[0].bounds
        if not (geom_bounds[0] <= cluster_bounds.right and
                geom_bounds[2] >= cluster_bounds.left and
                geom_bounds[1] <= cluster_bounds.top and
                geom_bounds[3] >= cluster_bounds.bottom):
            print("Error: Geometry does not overlap with cluster raster extent")
            return

        # Load deforestation raster metadata
        with rasterio.open(deforestation_tif_path) as def_src:
            def_crs = def_src.crs
            def_transform = def_src.transform
            def_pixel_size = (def_transform[0], -def_transform[4])
            print("Deforestation pixel size:", def_pixel_size)

        # Resolution check (updated to allow 25m)
        if not (abs(cluster_pixel_size[0] - 25) < 0.1 and abs(cluster_pixel_size[1] - 25) < 0.1):
            print(f"Warning: K-means clusters raster does not have 25m resolution, found {cluster_pixel_size}")
        if not (abs(def_pixel_size[0] - 25) < 0.1 and abs(def_pixel_size[1] - 25) < 0.1):
            print(f"Warning: Deforestation raster does not have 25m resolution, found {def_pixel_size}")

        # CRS check and reprojection if needed
        if cluster_crs != def_crs:
            # print(f"Warning: CRS mismatch. Clusters: {cluster_crs}, Deforestation: {def_crs}. Reprojecting deforestation raster.")
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                temp_defor_path = tmp.name
            reproject_raster(deforestation_tif_path, temp_defor_path, cluster_crs)
            print(deforestation_tif_path)
            deforestation_tif_path = temp_defor_path  # Use reprojected path
            print(deforestation_tif_path)

        # Mask cluster raster with geometry
        geom = [gdf.geometry.iloc[0].__geo_interface__]
        with rasterio.open(clusters_tif_path) as cluster_src:
            masked_clusters, masked_transform = mask(
                cluster_src, geom, crop=True, nodata=cluster_nodata
            )
            masked_clusters = masked_clusters[0]

        # Resample deforestation to match masked clusters
        deforestation_subset, def_nodata = match_resolution_and_extent(
            deforestation_tif_path, masked_clusters.shape, masked_transform, cluster_crs
        )

        # Calculate total deforestation per cluster
        cluster_deforestation = defaultdict(float)
        unique_clusters = np.unique(masked_clusters[~np.isclose(masked_clusters, cluster_nodata)])
        # print(f"Unique clusters found in masked area: {unique_clusters.tolist()}")

        for cluster_id in unique_clusters:
            cluster_mask = masked_clusters == cluster_id  # Changed variable name from 'mask' to 'cluster_mask'
            if def_nodata is not None:
                valid_deforestation = np.where(deforestation_subset != def_nodata, deforestation_subset, 0)
            else:
                valid_deforestation = deforestation_subset
            total_deforestation = np.sum(valid_deforestation[cluster_mask])  # Changed to use 'cluster_mask'
            cluster_deforestation[int(cluster_id)] = total_deforestation
            # print(f"Cluster {cluster_id}: Deforestation = {total_deforestation:.2f} ha/year, Pixels = {np.sum(cluster_mask)}")  # Changed to use 'cluster_mask'

        # Ensure clusters 3 and 4 are included, even if they have zero deforestation
        expected_clusters = set(cluster_deforestation.keys()).union({3, 4})
        results = []
        for cluster_id in sorted(expected_clusters):
            total = cluster_deforestation.get(cluster_id, 0.0)
            results.append({
                'cluster_id': int(cluster_id),
                'total_deforestation': float(total)
            })
            if cluster_id not in cluster_deforestation:
                # print(f"Cluster {cluster_id}: Deforestation = 0.00 ha/year, Pixels = 0")
                pass

        # Save to CSV
        df = pd.DataFrame(results)
        output_csv = output_pth
        df.to_csv(output_csv, index=False)
        print(f"Results saved to '{output_csv}'")

    except FileNotFoundError as e:
        print(f"Error: One or more files not found. Check paths: {clusters_tif_path}, {deforestation_tif_path}, {kmz_path}")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
    finally:
        # Clean up temporary file if it exists
        if temp_defor_path and os.path.exists(temp_defor_path):
            try:
                os.remove(temp_defor_path)
            except:
                pass

def is_within_bounds(x, y, bounds):
    return bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top