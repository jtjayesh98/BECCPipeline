import ee
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor


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