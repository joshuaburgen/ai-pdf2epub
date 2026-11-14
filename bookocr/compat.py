"""Scoped workaround for malformed certificates in the Windows system store."""
import ssl
from contextlib import contextmanager


@contextmanager
def certificate_fallback():
    original = ssl.create_default_context
    original_load = ssl.SSLContext.load_default_certs

    def load(context, *args, **kwargs):
        try:
            return original_load(context, *args, **kwargs)
        except ssl.SSLError as exc:
            if "NOT_ENOUGH_DATA" not in str(exc):
                raise
            import certifi
            return context.load_verify_locations(cafile=certifi.where())

    def create(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except ssl.SSLError as exc:
            if "NOT_ENOUGH_DATA" not in str(exc) or any(kwargs.get(k) for k in ("cafile", "capath", "cadata")):
                raise
            import certifi
            # Explicit CA bundle keeps hostname/certificate verification enabled.
            return original(*args, **kwargs, cafile=certifi.where())

    ssl.create_default_context = create
    ssl.SSLContext.load_default_certs = load
    try:
        yield
    finally:
        ssl.create_default_context = original
        ssl.SSLContext.load_default_certs = original_load
