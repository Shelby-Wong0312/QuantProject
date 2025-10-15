import os
from typing import Optional, Dict

import boto3

_ssm = boto3.client("ssm")
_secrets = boto3.client("secretsmanager")
_cache: Dict[str, str] = {}


def _cache_get(key: str) -> Optional[str]:
    return _cache.get(key)


def _cache_set(key: str, value: str) -> None:
    _cache[key] = value


def get_param(name: str, *, decrypt: bool = True) -> Optional[str]:
    if not name:
        return None
    if (v := _cache_get(f"ssm::{name}")) is not None:
        return v
    try:
        resp = _ssm.get_parameter(Name=name, WithDecryption=decrypt)
        val = resp.get("Parameter", {}).get("Value")
        if val is not None:
            _cache_set(f"ssm::{name}", val)
        return val
    except Exception:
        return None


def get_secret(secret_id: str) -> Optional[str]:
    if not secret_id:
        return None
    if (v := _cache_get(f"sm::{secret_id}")) is not None:
        return v
    try:
        resp = _secrets.get_secret_value(SecretId=secret_id)
        val = resp.get("SecretString")
        if val is not None:
            _cache_set(f"sm::{secret_id}", val)
        return val
    except Exception:
        return None
