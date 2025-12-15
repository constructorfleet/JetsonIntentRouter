from service.app import create_app


def test_chat_endpoint_non_stream():
    app = create_app()
    client = app.test_client()

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "router",
            "messages": [{"role": "user", "content": "turn off the lights"}],
            "stream": False,
        },
    )

    assert resp.status_code == 200
    data = resp.get_json()
    assert "choices" in data
