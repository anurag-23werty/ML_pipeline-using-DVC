import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging 
import yaml
log_dirs = 'logs'
os.makedirs(log_dirs,exist_ok=True)
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dirs,'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)
def load_params(param_path:str) -> dict:
    try:
        with open(param_path,'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s',param_path)
        return params
    except FileNotFoundError:
        logger.error('Parameter file not found at %s',param_path)
        raise
    except yaml.YAMLError as e:
        logger.error('Error parsing YAML file at %s: %s',param_path,e)
        raise
def load_data(data_path:str)->pd.DataFrame:
    try:
        df = pd.read_csv("spam.csv")
        logger.debug('Data loaded from %s',data_path)
        return df
    except FileNotFoundError:
        logger.error('Data file not found at %s',data_path)
        raise
def pre_process_data(df:pd.DataFrame)->pd.DataFrame:
    try:
        keep_cols = ['v1','v2']
        df = df[keep_cols]
        df.rename(columns = {'v1': 'target', 'v2': 'text'}, inplace = True)
        logger.debug("Data pre-processing completed successfully.")
        return df
    except KeyError as e:
        logger.error("Error during data pre-processing: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during data pre-processing: %s", e)
        raise
def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str):
    try:
        raw_data_path = os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,'train.csv'),index=False)
        test_data.to_csv(os.path.join(raw_data_path,'test.csv'),index=False)
        logger.debug("Train and test data saved successfully at %s",raw_data_path)
    except Exception as e:
            logger.error("Error saving data: %s", e)
            raise
def main():
    try:
        params = load_params(param_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        data_path = "https://github.com/anurag-23werty/ML_pipeline-using-DVC/blob/main/spam.csv"
        df = load_data(data_path)
        final_df = pre_process_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logger.error("Data ingestion failed: %s", e)
        print(f'ERROR: {e}  ')
if __name__ == "__main__":
    main()