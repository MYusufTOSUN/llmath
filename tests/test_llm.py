from llmath import llm
from llmath.config import MODELS


class _Reply:
    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": "\\boxed{1}"}, "prompt_eval_count": 3, "eval_count": 2, "done_reason": "stop"}


def _sent(monkeypatch, key, **kwargs):
    payloads = []
    monkeypatch.setattr(llm.httpx, "post", lambda url, json, timeout: payloads.append(json) or _Reply())
    llm.chat(MODELS[key], [{"role": "user", "content": "1 kaçtır?"}], **kwargs)
    return payloads[0]


def test_greedy_models_keep_the_main_run_settings(monkeypatch):
    options = _sent(monkeypatch, "qwen3-8b")["options"]
    assert options == {"temperature": 0.0, "num_ctx": 8192, "num_predict": 4096}


def test_reasoning_models_sample_as_their_publishers_recommend(monkeypatch):
    payload = _sent(monkeypatch, "deepseek-r1-8b")
    assert payload["think"] is True
    assert payload["options"] == {"temperature": 0.6, "num_ctx": 8192, "num_predict": 7680, "top_p": 0.95,
                                  "seed": 2026_10_21}
    # the output can never push the question out of the window
    assert payload["options"]["num_predict"] < payload["options"]["num_ctx"]
    assert _sent(monkeypatch, "qwen3-8b-think")["options"]["top_k"] == 20
    # an explicit temperature wins, as the LLM judge uses it to keep its labels deterministic
    assert _sent(monkeypatch, "qwen3-8b-think", temperature=0.0)["options"]["temperature"] == 0.0


def test_sampling_is_recorded_without_empty_fields():
    assert llm.sampling(MODELS["qwen3-8b"]) == {"temperature": 0.0, "num_ctx": 8192, "max_tokens": 4096,
                                                "think": False}
    assert llm.sampling(MODELS["phi4-mini-reasoning"])["temperature"] == 0.6
