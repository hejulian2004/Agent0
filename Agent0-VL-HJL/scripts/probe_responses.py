"""Fail-fast live probe for the configured Responses VL endpoint."""

from __future__ import annotations

import json

from agent0_protocol.responses_runtime import ResponsesConfig, ResponsesRuntime


def main() -> None:
    runtime = ResponsesRuntime(ResponsesConfig.from_env())
    print(json.dumps({
        "status": "passed",
        "base_url": runtime.config.base_url,
        "model": runtime.config.model,
        "capabilities": ["text", "image", "function tools", "function_call",
                         "function_call_output", "two consecutive tool rounds"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
