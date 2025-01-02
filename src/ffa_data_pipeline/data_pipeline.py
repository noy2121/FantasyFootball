import os
import hydra
from pathlib import Path
from unidecode import unidecode

from ingestion import load_raw_data
from aggregation import aggregate_data
from src.utils.utils import load_dataframes, save_dataframe, ROOT_DIR


@hydra.main(config_path='../config', config_name='data_conf')
def main(cfg):
    raw_data_dir = os.path.join(ROOT_DIR, cfg.raw_data_dir)
    processed_dir = os.path.join(ROOT_DIR, cfg.processed_data_dir)

    # Data Ingestion & Cleaning
    raw_data = load_dataframes(raw_data_dir)
    for key, df in raw_data.items():
        for column in df.select_dtypes(include=["object"]).columns:
            df[column] = df[column].apply(lambda x: unidecode(x) if isinstance(x, str) else x)
        raw_data[key] = df

    # Data Aggregation
    processed_data = aggregate_data(raw_data)

    output_path = f'{processed_dir}/processed.csv'
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    save_dataframe(processed_data, output_path)
    print(f"Processed dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
