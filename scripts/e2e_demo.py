from __future__ import annotations

from docsite_updater.demo import run_demo


def main() -> None:
    result = run_demo()
    pull_requests = result["pull_requests"]
    if pull_requests:
        print(f"Created docsite PR: {pull_requests[0].url}")
    else:
        print("No docsite PR created.")


if __name__ == "__main__":
    main()
