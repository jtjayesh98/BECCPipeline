import ee
import geemap
import uuid
from google.colab import auth
import json
from google.colab import drive
import rasterio
from rasterio.warp import transform
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import glob
import os

from utils import maskS2clouds, calculateNDVI, applySNIC, applyKMeans, applySNIC_with_meanNDVI, calculate_deforestation_by_cluster, is_within_bounds


def extract_features_per_cluster(cluster_id, clusters, cluster_band, geometry, predictors):
    try:
        # Create mask for the cluster
        mask = clusters.select(cluster_band).eq(cluster_id)

        # Check if the cluster has any pixels in the geometry
        pixel_count_dict = mask.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=10,
            maxPixels=1e9
        )
        pixel_count = pixel_count_dict.get(cluster_band).getInfo() if pixel_count_dict.get(cluster_band) is not None else 0

        if pixel_count == 0:
            print(f"No pixels found for cluster {cluster_id}. Returning NaN.")
            return [np.nan, np.nan]

        # Apply mask to predictors
        masked_predictors = predictors.updateMask(mask)

        # Calculate mean EVI and slope
        stats = masked_predictors.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=10,
            maxPixels=1e9
        )

        evi_val = stats.get('EVI').getInfo() if stats.get('EVI') is not None else None
        slope_val = stats.get('SLOPE').getInfo() if stats.get('SLOPE') is not None else None

        print(f"Cluster {cluster_id}: Pixel count = {int(pixel_count)} EVI = {evi_val}, SLOPE = {slope_val}")
        return [evi_val if evi_val is not None else np.nan,
                slope_val if slope_val is not None else np.nan]

    except ee.EEException as e:
        print(f"Error processing cluster {cluster_id}: {e}")
        return [np.nan, np.nan]


def knn_impute(geometry, image, biomass_csv_path = '/content/cluster_biomass_summary.csv', output_csv_path = 'imputed_biomass_clusters.csv', cluster_tif = '/content/drive/My Drive/GEE_Exports_Adhapalli/Adhapalli_KMeans_Cluster.tif'):

    s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(geometry).filterDate('2020-01-01', '2020-12-31').filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    s2 = s2_collection.median()
    band_names = s2.bandNames().getInfo()
    if not all(band in band_names for band in ['B2', 'B4', 'B8']):
        print('Required bands (B2, B4, B8) not found in Sentinel-2 image.')
        raise ValueError("missing required bands")
    
    nir = s2.select('B8')
    red = s2.select('B4')
    blue = s2.select('B2')

    try:
        evi = s2.expression(
            '2.4 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)',
            {
                'NIR': nir,
                'RED': red,
                'BLUE': blue
            }
        ).rename('EVI')
    except ee.EEException as e:
        print("Error calculating EVI:", e)
        raise

    srtm = ee.Image('USGS/SRTMGL1_003')
    slope = ee.Terrain.slope(srtm).rename('SLOPE')

    predictors = evi.addBands(slope)

    predictors.bandNames().getInfo()

    clusters = image.select('KMeans_clusters')

    cluster_bands = clusters.bandNames().getInfo()
    print(f"Cluster image bands: {cluster_bands}")
    # Assume first band contains cluster values
    cluster_band = cluster_bands[0]
    # Get unique cluster values (approximate, may be slow for large images)
    unique_values = clusters.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=geometry,
        scale=10,
        maxPixels=1e9
    ).get(cluster_band).getInfo()
    print(f"Unique cluster values in Adhapalli: {list(unique_values.keys())}")


    
    cluster_ids = [0, 1, 2, 3, 4, 5, 6]
    features = []
    for cid in cluster_ids:
        try:
            feat = extract_features_per_cluster(cid, clusters, cluster_band, geometry, predictors)
            features.append(feat)
        except Exception as e:
            print(f"Error processing cluster {cid}: {e}")
            features.append([np.nan, np.nan])

    feature_df = pd.DataFrame(features, columns=['EVI', 'SLOPE'], index=cluster_ids)


    try:
        biomass_df = pd.read_csv(biomass_csv_path)
        biomass_df.set_index('Cluster_ID', inplace=True)
    except FileNotFoundError:
        print(f"Error: File '{biomass_csv_path}' not found.")
        raise

    data_df = feature_df.join(biomass_df, how='left')

    if data_df[['EVI', 'SLOPE']].isna().all().any():
        print("Error: Some clusters have no EVI or SLOPE values. Check clusters.tif or geometry.")
        raise ValueError("Missing features for some clusters")
    elif data_df[['EVI', 'SLOPE']].isna().any().any():
        print("Warning: Missing EVI or SLOPE values for some clusters. Imputation may be affected.")

    X = data_df[['EVI', 'SLOPE']].values
    y = data_df['Biomass_per_ha'].values


    
    # Drop rows with missing X values (EVI/SLOPE must be present for imputation)
    valid_mask = ~np.isnan(X).any(axis=1)

    # Split into known and missing biomass
    X_known = X[valid_mask & ~np.isnan(y)]
    y_known = y[valid_mask & ~np.isnan(y)]

    X_missing = X[valid_mask & np.isnan(y)]
    missing_indices = np.where(valid_mask & np.isnan(y))[0]

    # Fit k-NN regressor
    knn = KNeighborsRegressor(n_neighbors=3, weights='distance')
    knn.fit(X_known, y_known)

    # Predict missing y
    y_pred = knn.predict(X_missing)

    # Fill in missing values
    y_imputed = y.copy()
    y_imputed[missing_indices] = y_pred
    print(y_imputed)

    imputed_df = pd.DataFrame({
        'Cluster_ID': cluster_ids,
        'Biomass_per_ha': y_imputed
    })

    imputed_df.to_csv(output_csv_path, index=False)
    
    print(f"Imputed biomass saved to {output_csv_path}")

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

    clusters_tif_path = '/content/drive/My Drive/GEE_Exports_Adhapalli/Adhapalli_KMeans_Cluster.tif'
    deforestation_tif_path = '/content/drive/My Drive/GEE_exports_Dhenkanal/Acre_Adjucted_Density_Map_VP.tif'
    kmz_path = '/content/drive/My Drive/GEE_exports_Dhenkanal/odisha_dhenkanal.kmz'
    output_pth = 'real_def_per_cluster.csv'

    calculate_deforestation_by_cluster(clusters_tif_path, deforestation_tif_path, kmz_path, output_pth)

    geotiff_path = '/content/drive/My Drive/GEE_Exports_Adhapalli/Adhapalli_KMeans_Cluster.tif'  # Replace with your GeoTIFF file path
    
    csv_path = '/content/drive/My Drive/GEE_Exports_Adhapalli/ADHAPALI_biomass.csv'          # Your CSV file

    # Load GeoTIFF
    dataset = rasterio.open(geotiff_path)

    # Load CSV
    df = pd.read_csv(csv_path)

    # Define coordinate systems
    src_crs = 'EPSG:4326'  # CSV coordinates are in WGS84 (lat/long)
    dst_crs = dataset.crs  # GeoTIFF coordinate system

    # Extract coordinates from CSV
    lons = df['Plot Long']
    lats = df['Plot Lat']

    # Transform coordinates to GeoTIFF CRS
    xs, ys = transform(src_crs, dst_crs, lons, lats)

    # Get GeoTIFF bounds for failsafe check
    bounds = dataset.bounds

    # Assign cluster values to each point
    cluster_values = []
    for x, y in zip(xs, ys):
        if is_within_bounds(x, y, bounds):
            # Sample the cluster value from the GeoTIFF (assuming single-band raster)
            value = next(dataset.sample([(x, y)]))[0]
            cluster_values.append(value)
        else:
            # Failsafe: assign None for points outside the GeoTIFF
            cluster_values.append(None)

    # Add cluster column to DataFrame
    df['Cluster'] = cluster_values

    # Group by cluster and save to separate files
    for cluster, group in df.groupby('Cluster'):
        if cluster is not None:
            # Save each cluster group to a CSV file
            group.to_csv(f'cluster_{int(cluster)}.csv', index=False)
        else:
            # Optionally save points outside the GeoTIFF to a separate file
            group.to_csv('cluster_outside.csv', index=False)

    # Close the GeoTIFF dataset
    dataset.close()


    # Find all cluster CSV files
    cluster_files = glob.glob('cluster_*.csv')

    # Initialize list to store results
    results = []

    # Process each cluster file
    for file in cluster_files:
        # Extract cluster ID from filename (e.g., 'cluster_0.csv' -> 0)
        if os.path.basename(file).split('_')[1].split('.')[0].isnumeric():
            cluster_id = int(os.path.basename(file).split('_')[1].split('.')[0])

        # Read the CSV file
            df = pd.read_csv(file)

            # Sum the 'Total biomass' column
            total_biomass = df['Total biomass'].sum()
            total_area_m2 = df['Area'].sum() # Here we are assuming the area is in m2, you can change it later according to the new information/data available

            if total_area_m2 > 0:
                biomass_per_ha = total_biomass / (total_area_m2 / 10000)
            else:
                biomass_per_ha = None # Handle division by zero if needed

            # Append result
            results.append({'Cluster_ID': cluster_id, 'Total_Biomass': total_biomass, 'Total_Area_m2': total_area_m2, 'Biomass_per_ha': biomass_per_ha})

    # Create DataFrame from results
    results_df = pd.DataFrame(results)

    # Sort by Cluster_ID for clarity
    results_df = results_df.sort_values('Cluster_ID')

    # Save to CSV
    results_df.to_csv('cluster_biomass_summary.csv', index=False)

    knn_impute(geometry, image)

    # deforestation_per_cluster_real = 'real_def_per_cluster.csv'

    expected_def = 'deforestation_by_cluster.csv'

    real_df = pd.read_csv(output_pth)
    expected_df = pd.read_csv(expected_def)
    biomass_df = pd.read_csv('imputed_biomass_clusters.csv')

    # Compute the weighted difference directly
    real = real_df['total_deforestation']
    expected = expected_df['total_deforestation']
    biomass = biomass_df['Biomass_per_ha']

    weighted_diff_sum = ((expected-real) * biomass).sum()

    print("Additionality:", weighted_diff_sum)







