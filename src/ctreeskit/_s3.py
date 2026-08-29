"""Helpers backing the optional ``s3`` extra (reading from ``s3://`` paths)."""

from typing import Any, Tuple


def s3_client() -> Any:
    """A boto3 S3 client using the standard AWS credential chain.

    Raises
    ------
    ImportError
        If boto3 is not installed. It ships with the optional ``s3`` extra
        (``pip install 'ctreeskit[s3]'``).
    """
    try:
        import boto3
    except ImportError as e:
        raise ImportError(
            "Reading from S3 requires boto3, which ships with the optional "
            "'s3' extra: pip install 'ctreeskit[s3]'"
        ) from e
    return boto3.client("s3")


def split_s3_uri(uri: str) -> Tuple[str, str]:
    """Split ``s3://bucket/key`` into ``(bucket, key)``."""
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return bucket, key
