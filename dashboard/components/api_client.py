from typing import Any

import pandas as pd
import httpx

from brandparadigm.config.settings import get_settings

API_BASE_URL = get_settings().dashboard_api_base_url.rstrip("/")


class DashboardAPIError(RuntimeError):
    pass


def _request(method: str, path: str, **kwargs: Any) -> Any:
    try:
        response = httpx.request(method, f"{API_BASE_URL}{path}", **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        detail = getattr(exc.response, "text", "") if exc.response is not None else ""
        raise DashboardAPIError(f"BrandParadigm API request failed: {detail or exc}") from exc


def get_model_health() -> dict[str, Any]:
    return _request("GET", "/health", timeout=2)


def analyze_review(text: str, brand="Unknown", product="Unknown", source="Manual"):
    return _request("POST", "/analyze", json={"text": text, "brand": brand, "product": product, "source": source}, timeout=120)


def analyze_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    if "text" not in frame:
        raise ValueError("The uploaded CSV must contain a 'text' column.")
    reviews = []
    for row in frame.to_dict("records"):
        reviews.append({key: row.get(key, default) for key, default in {"text": "", "brand": "Unknown", "product": "Unknown", "source": "CSV Upload", "date": None}.items()})
    result = _request("POST", "/analyze_batch", json={"reviews": reviews}, timeout=600)
    return pd.DataFrame(result["results"])
