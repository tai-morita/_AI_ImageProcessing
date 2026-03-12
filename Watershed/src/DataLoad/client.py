from __future__ import annotations

import hashlib
from typing import Any

import requests

from .settings import DataLoadConfig


class ApiClient:
    """Thin wrapper around the annotation API."""

    def __init__(self, config: DataLoadConfig) -> None:
        self.config = config
        self.session = requests.Session()

    @property
    def cert(self) -> tuple[str, str]:
        return (str(self.config.client_crt), str(self.config.client_key))

    @property
    def verify(self) -> str:
        return str(self.config.ca_path)

    def get_catalogue_nodes(self) -> list[str]:
        response = self.session.get(
            f"{self.config.url_base}/catalogue/{self.config.catalogue_key}",
            cert=self.cert,
            verify=self.verify,
        )
        response.raise_for_status()
        data = response.json()
        return data["nodes"]

    def get_node_attachment_uid(self, node_id: str) -> str:
        response = self.session.get(
            f"{self.config.url_base}/node/{node_id}",
            cert=self.cert,
            verify=self.verify,
        )
        response.raise_for_status()
        return response.json()["attachment"]

    def download_attachment(self, attachment_uid: str) -> bytes:
        response = self.session.get(
            f"{self.config.url_base}/attach/{attachment_uid}",
            cert=self.cert,
            verify=self.verify,
        )
        response.raise_for_status()
        return response.content

    def find_label_attachment_uid(self, dataset_node_id: str) -> str:
        key = anonymize_id(
            ",".join(self.config.catalogue_tags) + ":" + dataset_node_id
        )
        response = self.session.post(
            f"{self.config.url_base}/node/search",
            cert=self.cert,
            verify=self.verify,
            json={"key": key},
        )
        response.raise_for_status()
        nodes: list[dict[str, Any]] = response.json()
        if not nodes:
            raise RuntimeError("Label node not found for selected dataset.")
        return nodes[0]["attachment"]


def anonymize_id(raw_id: str) -> str:
    return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()
