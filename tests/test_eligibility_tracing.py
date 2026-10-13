"""병렬 기업 판정이 부모 실행의 추적 정보를 유지하는지 확인한다."""
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableLambda

from agents import eligibility


def test_company_checks_remain_children_of_eligibility(monkeypatch):
    runs = []

    class Capture(BaseCallbackHandler):
        def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None,
                           tags=None, metadata=None, **kwargs):
            runs.append((run_id, parent_run_id, tags, metadata))

    company_check = RunnableLambda(lambda c: ({**c, "eligible": False, "stage": "Unknown",
                                              "reason": "test"}, {}))
    monkeypatch.setattr(eligibility, "_check_one", lambda c, snapshot: company_check.invoke(c))
    candidates = [{"name": "A", "region": "KR"}, {"name": "B", "region": "KR"}]
    result = RunnableLambda(eligibility.eligibility_node).invoke(
        {"raw_candidates": candidates, "registry": {}},
        config={"callbacks": [Capture()], "tags": ["supervisor"], "metadata": {"run_id": "test-run"}},
    )

    assert len(result["screened"]) == 2
    root = next(run for run in runs if run[1] is None)
    children = [run for run in runs if run[1] == root[0]]
    assert len(children) == 2
    assert all("supervisor" in tags and metadata["run_id"] == "test-run"
               for _, _, tags, metadata in children)
