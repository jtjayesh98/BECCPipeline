import os
import rasterio
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, accuracy_score, f1_score, recall_score, precision_score, roc_curve
# from cupy import asnumpy

from GEEManager import GEEManager

class ML_ForestRiskPredictor():
    def __init__(self, working_directory, start_year, state_name):
        """
        Initialize the ML_RiskMaps class.

        Args:
            working_directory (str): Path to the local directory for saving outputs.
            start_year (int): The year to predict forest state for.
            state_name (str): Name of the target region (e.g., 'Tripura').
        """
        self.working_directory = working_directory
        self.start_year = start_year
        self.state_name = state_name
        self.years = list(range(2000, start_year))
        print(f"Years used for training: {self.years}")

        # Initialize GEE and Drive manager
        self.gee_manager = GEEManager()

        # self.drive_folder_path = f'/content/drive/My Drive/GEE_exports_{self.state_name}'

        self.drive_folder_path = "./images/TIF/"


    def run_gee_operations(self):
        """
        Run Google Earth Engine tasks to export forest cover maps for the defined years.
        """
        self.gee_manager.create_forest_maps_and_export(
            state_name=self.state_name,
            years=self.years
        )
        print("GEE operations completed successfully.")

    def prepare_data(self):
        """
        List downloaded forest rasters from Drive and generate forest change maps.
        """
        files = os.listdir(self.drive_folder_path)
        print(files)
        self.gee_manager.generate_forest_change_maps(
            folder_pth=self.drive_folder_path,
            out_dir=self.working_directory,
            state_name=self.state_name,
            years=self.years
        )
        print(f"Forest change maps generated successfully.")

    def raster_to_array(self, raster_path):
        """
        Load a raster file into a NumPy array with metadata.

        Args:
            raster_path (str): Full path to raster file.

        Returns:
            tuple: (raster array, affine transform, metadata)
        """
        with rasterio.open(raster_path) as src:
            return src.read(1), src.transform, src.meta

    def train_regressor(self, X, y, model_type="random_forest"):
        """
        Train a regression model on input data.

        Args:
            X (np.array): Flattened input features.
            y (np.array): Target values.
            model_type (str): Model type ("random_forest" only currently supported).

        Returns:
            object: Trained model.
        """
        if model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=10, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        model.fit(X, y)
        dump(model, f'{model_type}_reg.model')
        return model

    def train_classifier(self, X, y, model_type="random_forest"):
        """
        Train a classifier model on input data.

        Args:
            X (np.array): Flattened input features.
            y (np.array): Target values.
            model_type (str): Model type ("random_forest" only currently supported).

        Returns:
            object: Trained model.
        """
        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=10, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        model.fit(X, y)
        dump(model, f'{model_type}_class.model')
        return model

    def create_deforestation_prediction_map(self, model_type='random_forest'):
        """
        Train a model on historical deforestation maps and predict future deforestation.

        Output is saved as a GeoTIFF raster for visualization and further analysis.
        """
        X = []
        y = []

        # Prepare training data
        for i in range(len(self.years) - 2):

            year1 = self.years[i]
            year2 = self.years[i+1]
            year3 = self.years[i+2]

            arr1_path = os.path.join(self.working_directory, f'forest_change_map_{year1}_{year2}.tif')
            arr2_path = os.path.join(self.working_directory, f"forest_change_map_{year2}_{year3}.tif")

            arr1, _, _ = self.raster_to_array(arr1_path)
            arr2, _, _ = self.raster_to_array(arr2_path)

            X.append(arr1.flatten())
            y.append(arr2.flatten())

        X = np.concatenate(X).reshape(-1, 1)
        y = np.concatenate(y)

        # Train the model
        model = self.train_regressor(X, y, model_type=model_type)
        print("Here")
        # Prepare prediction data
        latest_year1 = self.years[-2]
        latest_year2 = self.years[-1]

        latest_arr_path = os.path.join(self.working_directory, f'forest_change_map_{latest_year1}_{latest_year2}.tif')
        latest_arr, transform, meta = self.raster_to_array(latest_arr_path)
        predicted = model.predict(latest_arr.flatten().reshape(-1, 1))
        predicted_reshaped = predicted.reshape(latest_arr.shape)

        # Save predicted forest change map
        output_path = os.path.join(self.working_directory, f'predicted_forest_change_map_{latest_year2}_{self.start_year}.tif')
        meta.update(dtype=rasterio.float32)
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(predicted_reshaped.astype(rasterio.float32), 1)

        print(f"Predicted forest change map saved to {output_path}")

        # Evaluate the model on training data
        y_true = y.astype(np.int32)
        y_pred = model.predict(X).astype(np.int32)

        print("MAE:", mean_absolute_error(y_true, y_pred))
        print("MSE:", mean_squared_error(y_true, y_pred))
        print("MedAE:", median_absolute_error(y_true, y_pred))

    def create_forest_cover_prediction_map(self, model_type="random_forest"):
        """
        Train a classifier on forest cover maps and predict forest cover for start_year.

        This method is useful for anticipating future forest extent using classification models.
        """
        X = []
        y = []

        drive_folder_path = "./images/TIF/"
        # Prepare training data from forest cover rasters
        for year1, year2 in zip(self.years[:-1], self.years[1:]):
            file1 = os.path.join(drive_folder_path, f'{self.state_name}_{year1}.tif')
            file2 = os.path.join(drive_folder_path, f'{self.state_name}_{year2}.tif')

            arr1, _, _ = self.raster_to_array(file1)
            arr2, _, _ = self.raster_to_array(file2)

            X.append(arr1.flatten())
            y.append(arr2.flatten())

        X = np.concatenate(X).reshape(-1, 1)
        y = np.concatenate(y)
        
        # Train classifier
        model = self.train_classifier(X, y, model_type=model_type)
        print("Here")
        # Predict forest cover for the target year
        latest_year = self.years[-1]
        latest_arr_path = os.path.join(drive_folder_path, f'{self.state_name}_{latest_year}.tif')
        latest_arr, transform, meta = self.raster_to_array(latest_arr_path)
        predicted = model.predict(latest_arr.flatten().reshape(-1, 1))
        predicted_reshaped = predicted.reshape(latest_arr.shape)

        # Save predicted forest cover map
        output_path = os.path.join(self.working_directory, f'predicted_forest_cover_map_{latest_year}.tif')
        meta.update(dtype=rasterio.uint8)

        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(predicted_reshaped.astype(rasterio.uint8), 1)

        print(f"Predicted forest cover map saved to {output_path}")

        # Evaluate the model on training data
        y_pred = model.predict(X)
        print("Accuracy:", accuracy_score(y, y_pred))
        print("F1 Score:", f1_score(y, y_pred))
        print("Precision:", precision_score(y, y_pred))
        print("Recall:", recall_score(y, y_pred))

