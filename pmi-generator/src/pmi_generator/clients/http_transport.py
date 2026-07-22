from __future__ import annotations

import ssl
import urllib.request
from pathlib import Path


def build_url_opener(
    *,
    verify_ssl: bool,
    ca_file: Path | None,
    no_proxy: bool,
) -> urllib.request.OpenerDirector:
    context = ssl_context(verify_ssl=verify_ssl, ca_file=ca_file)
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}) if no_proxy else urllib.request.ProxyHandler(),
        urllib.request.HTTPSHandler(context=context),
    )


def ssl_context(verify_ssl: bool, ca_file: Path | None) -> ssl.SSLContext:
    if not verify_ssl:
        return ssl._create_unverified_context()
    if ca_file is not None:
        return ssl.create_default_context(cafile=str(ca_file))
    return ssl.create_default_context()
