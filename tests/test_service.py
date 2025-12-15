from service.app import create_app


def test_health_endpoint():
    app = create_app()
    client = app.test_client()

    resp = client.get("/healthz")
    assert resp.status_code == 200
