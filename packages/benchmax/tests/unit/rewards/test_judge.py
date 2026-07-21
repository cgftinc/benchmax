import pickle

import pytest

from benchmax.auth import InjectedAuth, StaticBearerAuth
from benchmax.rewards import Judge
from benchmax.rewards.judge import _parse_json_object


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"score": 1}', {"score": 1}),
        ('```json\n{"score": 1}\n```', {"score": 1}),
        ('<think>private</think>\n{"score": 1}', {"score": 1}),
        ('preface {"score": 1} trailing', {"score": 1}),
    ],
)
def test_parse_json_object(raw, expected):
    assert _parse_json_object(raw) == expected


@pytest.mark.parametrize("raw", ["", "not json", "[1, 2]", '{"score": }'])
def test_parse_json_object_rejects_invalid_output(raw):
    with pytest.raises(ValueError, match="valid JSON object"):
        _parse_json_object(raw)


def test_judge_validates_configuration():
    auth = StaticBearerAuth("token")
    with pytest.raises(ValueError, match="model"):
        Judge(model="", base_url="https://judge.test", auth=auth)
    with pytest.raises(ValueError, match="base_url"):
        Judge(model="m", base_url="", auth=auth)
    with pytest.raises(ValueError, match="auth_attempts"):
        Judge(model="m", base_url="https://judge.test", auth=auth, auth_attempts=0)
    with pytest.raises(TypeError, match="ModelAuth"):
        Judge(model="m", base_url="https://judge.test", auth=object())


def test_judge_with_injected_auth_is_pickleable():
    judge = Judge(
        model="m",
        base_url="https://judge.test/v1",
        auth=InjectedAuth("judge"),
    )
    assert pickle.loads(pickle.dumps(judge)) == judge


@pytest.mark.asyncio
async def test_request_json_closes_client(judge_factory):
    stub = judge_factory(['{"score": 1}'])
    payload, raw = await stub.judge.request_json("prompt", request_id="request")
    assert payload == {"score": 1}
    assert raw == '{"score": 1}'
    stub.clients[0].close.assert_awaited_once()
