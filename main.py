import pandas as pd
from GEEManager import GEEManager
from ML_DeforestationRiskPredictor import ML_DeforestationRiskPredictor
from ML_ForestRiskPredictor import ML_ForestRiskPredictor
from RiskMaps import RiskMaps
import argparse
from biomass import execute_biomass
import ee
from google.colab import drive
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="stat", help="Type of Modeling that is to be done; stat (statistical) or ml (machine learning based approach)")
    parser.add_argument("--start_year", required=True, help="Start Year of the Project")
    parser.add_argument("--mid_point", required=True, help="Mid-point of the project; used to build the baseline model")
    parser.add_argument("--end_year", required=True, help="End point of the project")
    parser.add_argument("--jurisdiction_level", required=True, default="state", help="State or District level assessment")
    parser.add_argument("--district_name", required=True, help="District in which the area lies")
    parser.add_argument("--state_name", required = True, help = "State in which the area lies")
    # parser.add_argument("--working_directory", required = True)

    args = parser.parse_args()
    ee.Authenticate()
    ee.Initialize(project="ee-mtpictd-dev")
    drive.mount('/content/drive', force_remount=True)
    
    if args.model_type == "stat":
        if args.jurisdiction_level == "district":
            engine = RiskMaps(f'/content/drive/My Drive/GEE_exports_{args.district_name}', args.start_year, args.mid_point, args.end_year, args.district_name)
            engine.perform_gee_operations()
            engine.run_wo_gee()
            execute_biomass()
        elif args.jurisdiction_level == "state":
            engine = RiskMaps(f'/content/drive/My Drive/GEE_exports_{args.state_name}', args.start_year, args.mid_point, args.end_year, args.state_name)
            engine.perform_gee_operations()
            engine.run_wo_gee()
            execute_biomass()
    else:
        if args.jurisdiction_level == "district":
            engine = ML_DeforestationRiskPredictor(1)
            engine.prepare_data()
            engine.create_deforestation_prediction_map()
        elif args.jurisdiction_level == "state":
            engine = ML_DeforestationRiskPredictor(2)
            engine.prepare_data()
            engine.create_deforestation_prediction_map()

    




