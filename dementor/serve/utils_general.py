from __future__ import annotations

import hashlib
import pickle
from typing import Optional


def hash_key(key) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def get_from_cache(key: str, env: lmdb.Environment) -> Optional[str]:
    with env.begin(write=False) as txn:
        hashed_key = hash_key(key)
        value = txn.get(hashed_key.encode())
    if value:
        return value.decode()
    return None


def save_to_cache(key: str, value: str, env: lmdb.Environment):
    with env.begin(write=True) as txn:
        hashed_key = hash_key(key)
        txn.put(hashed_key.encode(), value.encode())


def save_emb_to_cache(key: str, value, env: lmdb.Environment):
    with env.begin(write=True) as txn:
        hashed_key = hash_key(key)
        serialized_value = pickle.dumps(value)
        txn.put(hashed_key.encode(), serialized_value)


def get_emb_from_cache(key: str, env: lmdb.Environment):
    with env.begin(write=False) as txn:
        hashed_key = hash_key(key)
        serialized_value = txn.get(hashed_key.encode())
        if serialized_value is not None:
            value = pickle.loads(serialized_value)
            return value
        else:
            return None

