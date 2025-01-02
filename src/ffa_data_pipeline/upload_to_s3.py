import os

import hydra
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from src.utils.utils import ROOT_DIR


def upload_files_to_s3(local_dir, bucket_name, s3_prefix):
    """
    Uploads all files from a local directory to an S3 bucket.

    Args:
        local_dir (str): Path to the local directory containing files to upload.
        bucket_name (str): Name of the S3 bucket.
        s3_prefix (str): S3 prefix (subdirectory) to upload files to.
    """
    s3 = boto3.client('s3')

    try:
        for filename in os.listdir(local_dir):
            if filename.endswith('.csv'):  # Only process CSV files
                local_path = os.path.join(local_dir, filename)
                s3_path = os.path.join(s3_prefix, filename)

                try:
                    s3.upload_file(local_path, bucket_name, s3_path)
                    print(f"Uploaded {filename} to s3://{bucket_name}/{s3_path}")
                except Exception as e:
                    raise f"Failed to upload {filename} to s3://{bucket_name}/{s3_path}: {str(e)}"

    except (NoCredentialsError, PartialCredentialsError) as e:
        raise f"AWS credentials not found or incomplete. Please configure them: {str(e)}"
    except Exception as e:
        raise f"An unexpected error occurred: {str(e)}"


@hydra.main(config_path='../config', config_name='data_conf')
def upload_data(cfg):

    # Configuration
    LOCAL_DATA_DIR = os.path.join(ROOT_DIR, cfg.data_dir)
    BUCKET_NAME = cfg.bucket_name
    S3_PREFIX = cfg.raw_data_dir

    # Validate local directory
    if not os.path.exists(LOCAL_DATA_DIR):
        raise FileNotFoundError(f"Local directory does not exist: {LOCAL_DATA_DIR}")

    # Start upload process
    print("Starting upload process...")
    upload_files_to_s3(LOCAL_DATA_DIR, BUCKET_NAME, S3_PREFIX)
    print("Upload process completed.")


if __name__ == "__main__":
    upload_data()


