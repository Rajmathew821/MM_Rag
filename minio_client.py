from minio import Minio
from minio.error import S3Error
from config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, BUCKET_NAME
from pathlib import Path
from datetime import timedelta

class MinioClient:
    def __init__(self):
        self.client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )
        self.ensure_bucket()

    def ensure_bucket(self):
        if not self.client.bucket_exists(BUCKET_NAME):
            self.client.make_bucket(BUCKET_NAME)

    def upload(self, file_path):
        object_name = Path(file_path).name
        try:
            self.client.fput_object(BUCKET_NAME, object_name, file_path)
            return object_name
        except S3Error as e:
            raise RuntimeError(f"MinIO upload error: {str(e)}")

    def get_presigned_url(self, object_name, expiry_seconds=3600):
        try:
            return self.client.presigned_get_object(
                BUCKET_NAME, object_name, expires=timedelta(seconds=expiry_seconds)
            )
        except S3Error as e:
            raise RuntimeError(f"Could not get presigned url: {e}")

    def exists(self, object_name):
        try:
            self.client.stat_object(BUCKET_NAME, object_name)
            return True
        except S3Error:
            return False
