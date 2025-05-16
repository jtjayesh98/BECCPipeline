import os
import rasterio
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from GEEManager import GEEManager

class ML_DeforestationRiskPredictor:
    def __init__(self, working_directory, start_year, state_name):
        """
        Initialize the ML_RiskMaps class for predicting deforestation risk.

        Args:
            working_directory (str): Directory containing input rasters and where outputs will be saved.
            start_year (int): Year for which to predict deforestation risk (e.g., 2005 predicts 2004->2005).
            state_name (str): Name of the area of interest (e.g., 'Acre').
        """
        self.working_directory = working_directory
        self.start_year = start_year
        self.state_name = state_name
        self.years = list(range(2000, start_year))
        self.feature_files = []
        self.output_dir = os.path.join(working_directory, 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        print(f'Years used for training: {self.years}')

        # Initialize Google Earth Engine (GEE) and Drive manager
        self.gee_manager = GEEManager()

        self.drive_folder_path = "./images/TIF/"

    def prepare_data(self):
        """
        Prepare feature rasters: distance from forest edge, distance from settlements, slope.
        """
        # Compute slope from DEM
        # self.gee_manager.extract_dem(state_name=self.state_name)
        dem_path = os.path.join(self.drive_folder_path, f'{self.state_name}_DEM.tif')
        if not os.path.exists(dem_path):
            raise FileNotFoundError(f'DEM file not found at {dem_path}')
        slope_path = os.path.join(self.output_dir, f'slope.tif')
        self.gee_manager.compute_slope(
            dem_path=dem_path,
            output_path=slope_path)
        self.feature_files.append(slope_path)

        # Compute distance to settlements
        # self.gee_manager.export_settlement_map(state_name=self.state_name)
        settlement_path = os.path.join(self.drive_folder_path, f'settlement_binary_{self.state_name}.tif')
        if not os.path.exists(settlement_path):
            raise FileNotFoundError(f'Settlement map not found at {settlement_path}')
        settlement_dist_path = os.path.join(self.output_dir, f'settlement_distance.tif')
        self.gee_manager.compute_distance(
            raster_path=settlement_path,
            output_path=settlement_dist_path,
            target_value=0
        )
        self.feature_files.append(settlement_dist_path)

        # Compute distance to forest edge for each year
        for year in self.years:
            forest_path = os.path.join(self.drive_folder_path, f'{self.state_name}_{year}.tif')
            if not os.path.exists(forest_path):
                raise FileNotFoundError(f'Forest map not found at {forest_path}')
            dist_edge_path = os.path.join(self.output_dir, f'forest_edge_distance_{year}.tif')
            self.gee_manager.compute_distance(
                raster_path=forest_path,
                output_path=dist_edge_path,
                target_value=0
            )
            self.feature_files.append(dist_edge_path)

    def train_model(self, X, y):
        """
        Train a Random Forest Classifier.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Labels

        Returns:
            RandomForestClassifier: Trained model
        """
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        print("Model Trained")
        dump(model, 'RF_Classifier.model')
        return model

    def create_deforestation_prediction_map(self):
        """
        Train model and predict deforestation risk for start_year
        """
        # Load feature rasters
        X_list = []
        y_list = []

        for i in range(len(self.years) - 1):
            year1 = self.years[i]
            year2 = self.years[i+1]

            with rasterio.open(os.path.join(self.drive_folder_path, f'{self.state_name}_{year1}.tif')) as src1:
                forest_y1 = src1.read(1)
            with rasterio.open(os.path.join(self.drive_folder_path, f'{self.state_name}_{year2}.tif')) as src2:
                forest_y2 = src2.read(1)
                profile = src2.profile

            # # Label1: 1 if deforested, 0 if remains forest
            label = (forest_y1 == 1) & (forest_y2 == 0)
            forest_mask = (forest_y1 == 1) # Only consider forested pixels
            label = label[forest_mask].astype(int)

            # Features for year1
            dist_edge_path = os.path.join(self.output_dir, f'forest_edge_distance_{year1}.tif')
            with rasterio.open(dist_edge_path) as src:
                dist_edge = src.read(1)[forest_mask]
            with rasterio.open(self.feature_files[0]) as src: # Slope
                slope = src.read(1)[forest_mask]
            with rasterio.open(self.feature_files[1]) as src: # Settlement distance
                dist_settlement = src.read(1)[forest_mask]

            X = np.stack([dist_edge, dist_settlement, slope], axis=1)
            X_list.append(X)
            y_list.append(label)

        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        print("Here")
        # Train model
        model = self.train_model(X, y)

        # Predict for start_year
        year_last = self.years[-1]
        with rasterio.open(os.path.join(self.drive_folder_path, f'{self.state_name}_{year_last}.tif')) as src:
            forest_last = src.read(1)

        dist_edge_path = os.path.join(self.output_dir, f'forest_edge_distance_{year_last}.tif')
        with rasterio.open(dist_edge_path) as src:
            dist_edge = src.read(1)

        with rasterio.open(self.feature_files[0]) as src: # Slope
            slope = src.read(1)

        with rasterio.open(self.feature_files[1]) as src: # Settlement distance
            dist_settlement = src.read(1)

        # Predict only for forested pixel
        forest_mask = (forest_last == 1)
        X_pred = np.stack([dist_edge[forest_mask], dist_settlement[forest_mask], slope[forest_mask]], axis=1)
        prob = np.zeros_like(forest_last, dtype=np.float32)
        if X_pred.size > 0:
            prob[forest_mask] = model.predict_proba(X_pred)[:, 1] # Probability of deforestation

        # Save prediction
        output_path = os.path.join(self.output_dir, f'deforestation_risk_{self.start_year}.tif')
        profile.update(dtype=rasterio.float32, count=1)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(prob, 1)

        print(f"Risk Map saved to {output_path}")

        # Evaluate on training data
        y_pred = model.predict(X)
        print(f"Training Accuracy: {accuracy_score(y, y_pred):.4f}")
        print("F1 Score:", f1_score(y, y_pred))
        print("Precision:", precision_score(y, y_pred))
        print("Recall:", recall_score(y, y_pred))