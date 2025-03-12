from arraylake import Client
from .common import ArraylakeDatasetConfig
from typing import Optional

DEFAULT_BUCKET = "arraylake-datasets"


class ArraylakeRepoCreator:
    """
    A class to simplify the creation and initialization of Arraylake repositories.

    This class supports two workflows:
      1. Direct creation using explicit parameters.
      2. Automated creation by processing JSON configuration files stored in S3.

    It uses the Arraylake API client to create repositories and boto3 to interact with S3.
    """

    def __init__(
        self,
        token: str,
        bucket_nickname: str = DEFAULT_BUCKET
    ):
        """
        Initialize the ArraylakeRepoCreator with an API token and S3 bucket nickname.

        Parameters
        ----------
        token : str
            Arraylake API token used for authenticating API requests.
        bucket_nickname : str, optional
            Nickname for the S3 bucket containing configuration files (default is "arraylake-datasets").
        """
        self.client = Client(token=token)
        self.bucket_nickname = bucket_nickname

    def create(self, dataset_name: str, organization_name: str) -> None:
        """
        Create an Arraylake repository using explicit dataset and organization parameters.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset for which the repository will be created.
        organization_name : str
            The name of the organization owning the dataset.

        Raises
        ------
        ValueError
            If dataset_name is empty.
        """
        if not dataset_name:
            raise ValueError("dataset_name is required for direct creation")

        repo_name = f"{organization_name}/{dataset_name}"
        self.client.create_repo(
            name=repo_name,
            bucket_config_nickname=self.bucket_nickname,
            kind="icechunk"
        )

    def create_from_s3(self, uri: Optional[str] = None) -> None:
        """
        Create repositories by processing JSON configuration files stored in S3.

        If a specific URI is provided, this method processes that single config file.
        Otherwise, it lists and processes all JSON files found under the "configs/" prefix in
        the default S3 bucket.

        Parameters
        ----------
        uri : Optional[str]
            An optional S3 URI for a specific JSON configuration file. If None, all config files in the bucket are processed.
        """
        if uri:
            # Process a single configuration file.
            self._process_config(uri)
        else:
            # Use the ArraylakeDatasetConfig class to list available datasets
            config_util = ArraylakeDatasetConfig()
            datasets = config_util.list_datasets()
            for dataset in datasets:
                uri = f"s3://{config_util.bucket}/{config_util.config_prefix}{dataset}.json"
                self._process_config(uri)

    def _process_config(self, uri: str) -> None:
        """
        Process a single JSON configuration file from S3 and create a repository if needed.

        The method extracts the dataset name from the URI, loads the configuration using
        ArraylakeDatasetConfig, and checks if a repository with the corresponding name already exists.
        If not, it creates a new repository.

        Parameters
        ----------
        uri : str
            The S3 URI of the JSON configuration file.

        Side Effects
        ------------
        Creates a repository via the Arraylake API client if one does not already exist.
        Prints the operation status (existing or created repository).
        """
        try:
            # Extract dataset name from the URI.
            config = ArraylakeDatasetConfig().load_config(s3_uri=uri)
            if not config.dataset_name:
                print(f"Missing dataset_name in config: {uri}")
                return

            try:
                # Check if repository already exists.
                self.client.get_repo(config.repo_name)
                print(f"Repository already exists: {config.repo_name}")
            except Exception:
                # Repository does not exist; create a new one.
                print(f"Creating repository: {config.repo_name}")
                self.client.create_repo(
                    name=config.repo_name,
                    bucket_config_nickname=DEFAULT_BUCKET,
                    kind="icechunk",
                )

        except Exception as e:
            print(f"Error processing config {uri}: {e}")
