"""HTTP client for the API. The UI talks only to this; it never sees the NVIDIA key."""

import httpx


class ApiError(Exception):
    """Any failure talking to the API, with a message fit to show the user."""


class RateLimited(ApiError):
    pass


class ApiClient:

    def __init__(self, base_url: str, timeout: float = 300.0, transport: httpx.BaseTransport | None = None):
        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, transport=transport)
        self.base_url = base_url

    def _request(self, method: str, path: str, **kwargs):
        try:
            resp = self._http.request(method, path, **kwargs)
        except httpx.TimeoutException as e:
            raise ApiError("The API took too long to answer. Try again, or lower the retry cap.") from e
        except httpx.HTTPError as e:
            raise ApiError(f"Cannot reach the API at {self.base_url}. Is it running?") from e
        if resp.status_code == 429:
            raise RateLimited(self._detail(resp) or "Rate limit reached. Wait a minute and try again.")
        if resp.status_code >= 400:
            raise ApiError(f"API error {resp.status_code}: {self._detail(resp) or resp.reason_phrase}")
        return resp.json()

    @staticmethod
    def _detail(resp: httpx.Response) -> str:
        try:
            detail = resp.json().get("detail", "")
        except ValueError:
            return ""
        return detail if isinstance(detail, str) else str(detail)

    def health(self) -> bool:
        try:
            return self._request("GET", "/health").get("status") == "ok"
        except ApiError:
            return False

    def settings(self) -> dict:
        return self._request("GET", "/settings")

    def ask(self, question: str, settings: dict | None = None) -> dict:
        return self._request("POST", "/ask", json={"question": question, "settings": settings or {}})

    def upload_documents(self, files: list[tuple[str, bytes]]) -> list[dict]:
        """files: (filename, content). Returns one result per file: added / already uploaded / rejected."""
        return self._request("POST", "/documents", files=[("files", (name, data)) for name, data in files])

    def list_documents(self) -> list[dict]:
        return self._request("GET", "/documents")

    def delete_document(self, doc_id: str) -> dict:
        return self._request("DELETE", f"/documents/{doc_id}")
