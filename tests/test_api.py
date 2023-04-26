"""Basic API tests."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ichthywhat.api import api


@pytest.fixture()
def api_client() -> TestClient:
    """Return a TestClient for the API."""
    return TestClient(api)


def test_home(api_client: TestClient) -> None:
    """Test the / endpoint."""
    response = api_client.get("/")
    assert response.status_code == 200
    assert response.text == '"Hello!"'


def test_demo(api_client: TestClient) -> None:
    """Test the /demo endpoint."""
    response = api_client.get("/demo")
    assert response.status_code == 200
    assert "<!DOCTYPE html>" in response.text


def test_predict(api_client: TestClient) -> None:
    """Test the /predict endpoint."""
    with (Path(__file__).parent / "pterois-volitans.jpg").open("rb") as fp:
        response = api_client.post("/predict", files={"img_file": fp})
    assert response.status_code == 200
    assert len(response.json()) > 0
    expected_first_name = "Pterois volitans"
    sum_scores = 0.0
    for i, (name, score) in enumerate(response.json().items()):
        if not i:
            assert expected_first_name == name
        assert 0.0 <= score <= 1.0
        sum_scores += score
    assert pytest.approx(1.0) == sum_scores
