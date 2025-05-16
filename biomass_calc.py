import ee
import geemap
import uuid
from google.colab import auth
import json
from google.colab import drive

from utils import maskS2clouds, calculateNDVI, applySNIC, applyKMeans, applySNIC_with_meanNDVI

def execute(state_name, district_name, site_name):
    ee.Authenticate()
    ee.Initialize(project="ee-mtpictd-dev")

    drive.mount('/content/drive', force_remount=True)

    try:
        state_sites = ee.FeatureCollection('projects/ee-mtpictd-dev/assets/' + state_name + '_sites')
    except ee.EEException as e:
        print("Error accessing asset. Check asset path or permissions:", e)
        raise

    # Filter for Adhapalli
    site = state_sites.filter(ee.Filter.eq('Name', site_name))

    # Check if Adhapalli exists
    num_site = site.size().getInfo()
    if num_site == 0:
        print("No feature found with Name = '" + site_name + "'. Check the Name value or asset.")
        raise ValueError(site_name + " not found")
    elif num_site > 1:
        print(f"Warning: {num_site} features found with Name = '" + site_name + "'. Using the first one.")
    print(f"Number of " + site_name + " features: {num_site}")

    # Extract geometry of the first Adhapalli feature
    site_feature = site.first()
    site_geometry = site_feature.geometry()

    # Get geometry coordinates as GeoJSON
    geojson = site_geometry.getInfo()
    # print("Adhapalli polygon geometry (GeoJSON):")
    # geojson = json.dumps(geojson, indent=2)

    geometry = ee.Geometry.Polygon([geojson['coordinates']])


    dataset = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterDate('2020-01-01', '2020-01-30')
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
           .filterBounds(geometry)
           .map(maskS2clouds)
           .map(calculateNDVI)
           .map(applySNIC)
           .map(lambda img: applyKMeans(img, num_clusters=7)))
    
    dataset = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate('2020-01-01', '2020-01-30')
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            .filterBounds(geometry)
            .map(maskS2clouds)
            .map(calculateNDVI)
            .map(applySNIC_with_meanNDVI)
            .map(lambda img: applyKMeans(img, num_clusters=7)))
    
    image = dataset.first().clip(geometry)
    # Export to Google Drive
    export_task = ee.batch.Export.image.toDrive(
        image=image.select(['KMeans_clusters']),
        description=site_name + '_Sentinel2_SNIC_202001',
        folder='GEE_Exports_' + site_name,
        fileNamePrefix=site_name + '_KMeans_Cluster',
        region=geometry,
        scale=25,
        maxPixels=1e9,
        fileFormat='GeoTIFF'
    )

    export_task.start()
    

