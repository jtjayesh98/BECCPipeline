import pandas as pd
import numpy as np
import math
import re
import zipfile
from xml.dom import minidom
from typing import List, Tuple

def load_project_areas(kmz_path: str) -> List[str]:
    """
    Extracts placemark names from a KMZ file and returns them in uppercase.

    Args:
        kmz_path (str): Path to the KMZ file.

    Returns:
        List[str]: List of placemark names in uppercase.
    """
    with zipfile.ZipFile(kmz_path, 'r') as kmz:
        kml_file = [f for f in kmz.namelist() if f.endswith('.kml')][0]
        kml_data = kmz.read(kml_file)

    doc = minidom.parseString(kml_data)
    placemarks = doc.getElementsByTagName('Placemark')
    return [p.getElementsByTagName("name")[0].firstChild.nodeValue.upper() for p in placemarks]


def preprocess_odisha_data(excel_path: str, district: str, project_areas: List[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Loads and preprocesses Odisha vegetation data, filtering by district and splitting into villages and panchayats.

    Args:
        excel_path (str): Path to the Excel file.
        district (str): District name to filter.
        project_areas (List[str]): List of project areas from KMZ.

    Returns:
        Tuple[pd.DataFrame, List[str], List[str]]: Filtered DataFrame, list of villages, list of panchayats.
    """
    df = pd.read_excel(excel_path)
    df = df[df['Habdistrict'] == district]
    
    # Clean and calculate DBH
    df['DBH'] = df['Gbh Girth'] / math.pi
    df.dropna(subset=['DBH'], inplace=True)
    df['Scientific Name'] = df['Scientific Name'].str.replace('[-_]', ' ', regex=True)
    df = df[df['Species Available'] == 'Yes']
    
    # Identify villages and panchayats
    villages = [area for area in project_areas if area in df['Habrvillage'].unique() and area not in df['Habpanchayat'].unique()]
    panchayats = [area for area in project_areas if area in df['Habpanchayat'].unique()]
    
    return df, villages, panchayats


def preprocess_rajasthan_data(excel_path: str) -> pd.DataFrame:
    """
    Loads and preprocesses Rajasthan vegetation data, calculating DBH and cleaning scientific names.

    Args:
        excel_path (str): Path to the Excel file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with DBH and cleaned scientific names.
    """
    df = pd.read_excel(excel_path)
    df['DBH'] = df['GBH_GIRTH'] / math.pi
    df.dropna(subset=['DBH'], inplace=True)
    df['Scientific Name'] = df['SCIENTIFIC_NAME'].str.replace('-', ' ', regex=False)
    return df


def save_area_data(df: pd.DataFrame, state: str, villages: List[str], panchayats: List[str] = None) -> None:
    """
    Saves DataFrame subsets for each village and panchayat as CSV files.

    Args:
        df (pd.DataFrame): Input DataFrame.
        state (str): State name for folder path.
        villages (List[str]): List of village names.
        panchayats (List[str], optional): List of panchayat names.
    """
    if state == "Odisha":
        for village in villages:
            df[df['Habrvillage'] == village].to_csv(f'Data/{state}/{village}.csv', index=False)
        if panchayats:
            for panchayat in panchayats:
                df[df['Habpanchayat'] == panchayat].to_csv(f'Data/{state}/{panchayat}.csv', index=False)
    elif state == "Rajasthan":
        for village in villages:
            df[df['HABRVILLAGE'] == village].to_csv(f'Data/{state}/{village}.csv', index=False)
        if panchayats:
            for panchayat in panchayats:
                df[df['HABPANCHAYAT'] == panchayat].to_csv(f'Data/{state}/{panchayat}.csv', index=False)        



def select_allometric_equations(village, state, eq_csv_path='Data/allometric_equations.csv'):
    # Load cleaned Adhapali CSV
    df = pd.read_csv(f'Data/{state}/{village}.csv')
    
    # Load allometric equations CSV
    eq_df = pd.read_csv('Data/allometric_equations.csv')
    
    # Clean up
    df['Scientific Name'] = df['Scientific Name'].str.strip()
    eq_df['Scientific_name'] = eq_df['Scientific_name'].str.strip()

    # Filter rows with valid binomial scientific names
    # Step 1: Remove NaN, None, or empty strings
    df = df[df['Scientific Name'].notna() & (df['Scientific Name'] != '')]
    
    # Step 2: Remove rows with single words or invalid values
    invalid_values = ['0', 'unidentified', 'unknown', 'not identified']  # Add more as needed
    df = df[
        # At least two words (genus + species)
        (df['Scientific Name'].str.split().str.len() >= 2) &
        # Not in invalid values (case-insensitive)
        (~df['Scientific Name'].str.lower().isin([val.lower() for val in invalid_values]))
    ]
    
    # Columns for TRUE count
    priority_cols = ['B', 'Bd', 'Bg', 'Bt', 'L', 'S', 'T', 'F']
    
    # Default values
    default_values = {
        'U': np.nan,
        'Unit_U': np.nan,
        'V': np.nan,
        'Unit_V': np.nan,
        'W': np.nan,
        'Unit_W': np.nan,
        'X': 'DBH',
        'Unit_X': 'cm',
        'Z': np.nan,
        'Unit_Z': np.nan,
        'Output': 'Biomass',
        'Unit_Y': 'kg',
        'Equation': 'exp(-2.134+2.530*ln(X))'
    }
    
    # Prepare columns in df
    for col in default_values:
        df[col] = np.nan
    
    # Ensure boolean consistency for TRUE checking
    eq_df[priority_cols] = eq_df[priority_cols].applymap(lambda x: str(x).strip().lower() == 'true')
    
    # Process each row
    for i, row in df.iterrows():
        full_name = str(row['Scientific Name']).strip()
        species_epithet = ' '.join(full_name.split(' ')[1:]) if len(full_name.split(' ')) > 1 else ''
        
        # Match species by substring
        matches = eq_df[eq_df['Scientific_name'].str.contains(species_epithet, case=False, na=False)]
        
        # Matches to filter complex equations
        valid_inputs = {"DBH", "H", "WD"}
        matches = matches[
            (matches["U"].isin(valid_inputs) | matches["U"].isna()) &
            (matches["V"].isin(valid_inputs) | matches["V"].isna()) &
            (matches["W"].isin(valid_inputs) | matches["W"].isna()) &
            (matches["X"].isin(valid_inputs) | matches["X"].isna()) &
            (matches["Z"].isin(valid_inputs) | matches["Z"].isna())
        ]
    
        if not matches.empty:
            matches = matches.copy()
    
            # Step 1: TRUE count priority
            matches['true_count'] = matches[priority_cols].sum(axis=1)
            max_true = matches['true_count'].max()
            matches = matches[matches['true_count'] == max_true]
    
            # Step 2: Output preference
            if len(matches) > 1:
                biomass_matches = matches[matches['Output'].str.lower().str.contains('biomass', na=False)]
                if not biomass_matches.empty:
                    matches = biomass_matches
                # else:
                #     matches = matches[matches['Output'].str.lower().str.contains('volume', na=False)]
    
            # Step 3: R2
            if len(matches) > 1:
                if 'R2' in matches.columns and matches['R2'].notna().any():
                    matches = matches[matches['R2'] == matches['R2'].max()]
                # otherwise we just take the first row as best match
    
            best_match = matches.iloc[0]
            for col in default_values:
                df.at[i, col] = best_match[col] if col in best_match else np.nan
                
            # ✅ Add logging info
            # df.at[i, "Equation Match Log"] = (f"Matched (Row {best_match.name}): {best_match['Scientific_name']} | "f"Output: {best_match['Output']} | R2: {best_match.get('R2', 'NA')}")
        
        else:
            # species not found in allometric_equations.csv — use default
            for col, val in default_values.items():
                # df.at[i, col] = row['DBH'] if val == 'DBH' else val
                df.at[i, col] = val
                
            # ❗ Log fallback case
            # df.at[i, "Equation Match Log"] = "Default used"
    
    # Save updated file
    df.to_csv(f'Data/{state}/{village}_allometric.csv', index=False)





def biomass(village, state):
    # Load the processed Adhapali CSV with matched equations
    df = pd.read_csv(f"Data/{state}/{village}_allometric.csv")

    # Constants
    WD_CONST = 0.57  # g/cm³
    DBH_UNIT = 'cm'
    H_UNIT = 'm'

    # Default biomass equation and its components
    DEFAULT_EQUATION = "exp(-2.134 + 2.530 * log(DBH))"

    # Normalize the Equation
    def normalize_equation(eq):
        if pd.isna(eq):
            return eq
        eq = eq.replace("^", "**")
        eq = re.sub(r'\b(log|LOG|Log)\b', 'log', eq)
        eq = re.sub(r'\b(ln|LN|Ln)\b', 'log', eq)
        eq = re.sub(r'\b(sqrt|SQRT|Sqrt)\b', 'sqrt', eq)
        eq = re.sub(r'\b(exp|EXP|Exp)\b', 'exp', eq)
        return eq

    # Helper function to extract variable names
    def extract_variables(equation):
        return re.findall(r'\b[UVWXYZ]\b', equation)

    # Get actual variable values from the row
    def get_variable_value(var, row):
        source = row[var]
        unit_col = f"Unit_{var}"
        unit = row[unit_col] if pd.notna(row[unit_col]) else None

        if source == "DBH":
            val = row["DBH"]
            if unit == "m":
                return val / 100
            return val
        elif source == "H":
            val = row["HEIGHT"]
            if unit == "cm":
                return val * 100
            return val
        elif source == "WD":
            return WD_CONST
        else:
            return np.nan

    # Default fallback biomass equation using DBH
    def evaluate_default_biomass(dbh):
        return math.exp(-2.134 + 2.530 * math.log(dbh))

    # Evaluate biomass for each row
    def evaluate_equation(row):
        eq = row["Equation"]
        if pd.isna(eq):
            return np.nan, None

        try:
            cleaned_eq = normalize_equation(eq)
            variables = extract_variables(cleaned_eq)
            env = {}
            for var in variables:
                env[var] = get_variable_value(var, row)

            if any(pd.isna(val) for val in env.values()):
                parsed = cleaned_eq
                for var in variables:
                    parsed = parsed.replace(var, str(env[var]))
                return np.nan, parsed

            parsed_eq = cleaned_eq
            for var, val in env.items():
                parsed_eq = parsed_eq.replace(var, str(val))

            result = eval(cleaned_eq, {
                "__builtins__": None,
                "exp": math.exp,
                "log": math.log,
                "sqrt": math.sqrt,
                "log10": math.log10
            }, env)

            # Adjust units
            output = str(row["Output"]).lower()
            unit_y = str(row["Unit_Y"]).lower()

            if "biomass" in output:
                if unit_y == "g":
                    result /= 1000
            elif "volume" in output:
                if unit_y == "cm3":
                    result *= WD_CONST / 1000
                elif unit_y == "m3":
                    result *= WD_CONST * 1000

            # Fallback to default equation if result is negative
            if result < 0 or result > 6000:
                dbh = row["DBH"]
                if pd.notna(dbh):
                    fallback_result = evaluate_default_biomass(dbh)
                    return fallback_result, DEFAULT_EQUATION.replace("DBH", str(dbh))
                else:
                    return np.nan, parsed_eq

            return result, parsed_eq

        except Exception as e:
            print(f"Error evaluating row {row.name}: {e}")
            return np.nan, None

    # Apply the evaluation function
    df[["Total biomass", "Parsed Equation"]] = df.apply(
        lambda row: pd.Series(evaluate_equation(row)),
        axis=1
    )

    # Save updated CSV
    df.to_csv(f"Data/{state}/{village}_biomass.csv", index=False)


def execute_biomass():
    """Main function to process data and calculate biomass for Odisha and Rajasthan."""
    # Odisha Processing
    odisha_areas = load_project_areas('Data/Odisha_Dhenkanal_21_9_2024.kmz')
    odisha_areas[0] = 'ADHAPALI'
    odisha_areas[-1] = 'KERAJOLI'
    
    df_odisha, villages_odisha, panchayats_odisha = preprocess_odisha_data(
        'Data/Odisha_data_24th April.ods', 'DHENKANAL', odisha_areas
    )
    save_area_data(df_odisha, 'Odisha', villages_odisha, panchayats_odisha)
    
    # Rajasthan Processing
    df_rajasthan = preprocess_rajasthan_data('Data/vegetation_data_Rajasthan.xlsx')
    villages_rajasthan = [
        'KADECH', 'MAKRADEO', 'Khardiya', 'DHIKORA','DHEEMRI', 'Dheemri',
        'KANJI KA GUDA', 'THARIVERI', 'Nalachhota', 'UPLI SIGARI', 'ROP',
        'SULTANJI KA KHERWARA'
    ]
    save_area_data(df_rajasthan, 'Rajasthan', villages_rajasthan)
    
    # Apply allometric equations and calculate biomass
    for village in villages_odisha + panchayats_odisha:
        select_allometric_equations(village, 'Odisha')
        biomass(village, 'Odisha')
    
    for village in villages_rajasthan:
        print(village)
        select_allometric_equations(village, 'Rajasthan')
        biomass(village, 'Rajasthan')
    print("Processing complete.")