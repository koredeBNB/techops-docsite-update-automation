from __future__ import annotations

from docsite_updater.demo import run_demo


def test_e2e_demo_creates_docsite_pr_for_relevant_source_change() -> None:
    result = run_demo(diff="+ API endpoint get_validator now returns validator status")

    assert result["webhook_status"] == "enqueued"
    assert result["jobs_remaining"] == 0
    assert result["dead_letters"] == 0
    assert len(result["pull_requests"]) == 1
    pr = result["pull_requests"][0]
    assert pr.url == "https://github.com/bnb-chain/mock-bnb-docsite/pull/1"
    assert pr.changed_files == ["docs/api.md"]
    assert "Automated Update" in result["doc_files"]["docs/api.md"]
    assert result["metrics"].counters["docs_pr.created"] == 1


def test_e2e_demo_creates_no_docsite_pr_when_no_doc_change_needed() -> None:
    result = run_demo(diff="NO_DOC_CHANGE: internal implementation cleanup")

    assert result["webhook_status"] == "enqueued"
    assert result["jobs_remaining"] == 0
    assert result["dead_letters"] == 0
    assert result["pull_requests"] == []
    assert result["metrics"].counters["job.no_changes_needed"] == 1
