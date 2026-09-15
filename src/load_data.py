from pathlib import Path
import pandas as pd


FILES = {
    "gen1": "Plant_1_Generation_Data.csv",
    "sensor1": "Plant_1_Weather_Sensor_Data.csv",
    "gen2": "Plant_2_Generation_Data.csv",
    "sensor2": "Plant_2_Weather_Sensor_Data.csv"
}


def load_raw(data_dir="data"):
    """
    Load the four original CSV files.

    DATE_TIME is intentionally kept as a string here.
    Date conversion is performed in prepare.py.
    """

    data_dir = Path(data_dir)

    frames = {}

    for key, filename in FILES.items():

        path = data_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path}\n"
                f"Please put the Kaggle CSV files inside the data folder."
            )

        frames[key] = pd.read_csv(path)

    return frames


if __name__ == "__main__":

    raw = load_raw()

    for key, df in raw.items():

        print("=" * 60)
        print(key)

        print("Rows:", df.shape[0])
        print("Columns:", df.shape[1])

        print("\nFirst 3 rows:")
        print(df.head(3).to_string())

        print("\nColumn names:")
        print(list(df.columns))

        print()