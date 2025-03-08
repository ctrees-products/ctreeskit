from arraylake import Client
import boto3
from botocore.exceptions import ClientError
from common import ArraylakeDatasetConfig
from typing import Optional

DEFAULT_BUCKET = "arraylake-datasets"


class ArraylakeRepoCreator:
    """A class to simplify creation of Arraylake repositories.

    This class handles creation in two ways:
    1. Direct creation with explicit parameters
    2. Creation from S3 config files
    """

    def __init__(
        self,
        token: str,
    ):
        """Initialize RepoCreator

        Args:
            token (str): Arraylake API token
        """
        self.client = Client(token=token)

    def create(self, dataset_name: str, organization_name: str, bucket_nickname: str = DEFAULT_BUCKET) -> None:
        """Create repository with explicit parameters."""
        if not dataset_name:
            raise ValueError("dataset_name is required for direct creation")

        self.client.create_repo(
            name=f"{organization_name}/{dataset_name}",
            bucket_config_nickname=bucket_nickname,
            kind="icechunk"
        )

    def create_from_s3(self, uri: Optional[str] = None) -> None:
        """Create repositories from S3 JSON configs.

        Args:
            uri: Optional specific JSON config path. If None, processes all JSONs in configs dir.
        """
        s3 = boto3.client('s3')

        if uri:
            # Process single config
            self._process_config(uri)
        else:
            # List and process all configs
            try:
                bucket = DEFAULT_BUCKET
                prefix = "configs/"
                response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

                for obj in response.get('Contents', []):
                    if obj['Key'].endswith('.json'):
                        uri = f"s3://{bucket}/{obj['Key']}"
                        self._process_config(uri)
            except ClientError as e:
                print(f"Error accessing S3: {e}")

    def _process_config(self, uri: str) -> None:
        """Process a single config file and create repository if needed."""
        try:
            dataset_name = uri.split('/')[-1].replace('.json', '')
            config = ArraylakeDatasetConfig(dataset_name)

            if not config.dataset_name:
                print(f"Missing dataset_name in config: {uri}")
                return

            try:
                self.client.get_repo(config.repo_name)
                print(f"Repository already exists: {config.repo_name}")
            except:
                print(f"Creating repository: {config.repo_name}")
                self.client.create_repo(
                    name=config.repo_name,
                    bucket_config_nickname=DEFAULT_BUCKET,
                    kind="icechunk",
                )

        except Exception as e:
            print(f"Error processing config {uri}: {e}")
