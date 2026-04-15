
import contextlib
import math
import os
import queue
import re
import sys
import subprocess
import time
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from jupyter_client import KernelManager
except ImportError:
    KernelManager = None

try:
    from openai_harmony import (
        Author,
        Conversation,
        HarmonyEncodingName,
        Message,
        ReasoningEffort,
        Role,
        SystemContent,
        TextContent,
        ToolNamespaceConfig,
        load_harmony_encoding,
    )
except ImportError:
    load_harmony_encoding = None


# ============================================================
# Configuration
# ============================================================

@dataclass
class CFG44:
    system_prompt: str = (
        "You are a world-class International Mathematical Olympiad (IMO) competitor. "
        "The final answer must be a non-negative integer between 0 and 99999. "
        "You must place the final integer answer inside \\boxed{}."
    )
    tool_prompt: str = (
        "Use this tool to execute Python code. "
        "The environment is a stateful Jupyter notebook. "
        "You must use print() to output results."
    )
    preference_prompt: str = (
        "You have access to `math`, `numpy` and `sympy` to solve the problem."
    )
    served_model_name: str = "openai/gpt-oss-120b"
    model_path: str = "/kaggle/input/models/danielhanchen/gpt-oss-120b/transformers/default/1"
    dtype: str = "auto"
    high_problem_timeout: float = 900
    base_problem_timeout: float = 300
    notebook_limit: float = 17400
    server_timeout: int = 180
    session_timeout: float = 960
    jupyter_timeout: float = 6
    sandbox_timeout: float = 3
    context_tokens: int = 65536
    buffer_tokens: int = 512
    search_tokens: int = 32
    top_logprobs: int = 5
    early_stop: int = 4
    attempts: int = 8
    workers: int = 8
    turns: int = 128
    seed: int = 42
    gpu_memory_utilization: float = 0.95
    temperature: float = 1.0
    min_p: float = 0.02


# ============================================================
# Jupyter Sandbox
# ============================================================

class AIMO3Sandbox:
    _port_lock = threading.Lock()
    _next_port = 50000

    @classmethod
    def _get_next_ports(cls, count=5):
        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + count))
            cls._next_port += count
            return ports

    def __init__(self, timeout):
        self._default_timeout = timeout
        self._owns_kernel = False
        self._client = None
        self._km = None
        ports = self._get_next_ports(5)
        env = os.environ.copy()
        env["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
        env["PYTHONWARNINGS"] = "ignore"
        env["MPLBACKEND"] = "Agg"
        self._km = KernelManager()
        self._km.shell_port = ports[0]
        self._km.iopub_port = ports[1]
        self._km.stdin_port = ports[2]
        self._km.hb_port = ports[3]
        self._km.control_port = ports[4]
        self._km.start_kernel(env=env, extra_arguments=["--Application.log_level=CRITICAL"])
        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._default_timeout)
        self._owns_kernel = True
        self.execute("import math\nimport numpy\nimport sympy\nimport itertools\nimport collections\nimport mpmath\nmpmath.mp.dps = 64\n")

    def execute(self, code, timeout=None):
        client = self._client
        effective_timeout = timeout or self._default_timeout
        msg_id = client.execute(code, store_history=True, allow_stdin=False, stop_on_error=False)
        stdout_parts, stderr_parts = [], []
        start_time = time.time()
        while True:
            if time.time() - start_time > effective_timeout:
                self._km.interrupt_kernel()
                return f"[ERROR] Execution timed out after {effective_timeout} seconds"
            try:
                msg = client.get_iopub_msg(timeout=1.0)
            except queue.Empty:
                continue
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            msg_type = msg.get("msg_type")
            content = msg.get("content", {})
            if msg_type == "stream":
                (stdout_parts if content.get("name") == "stdout" else stderr_parts).append(content.get("text", ""))
            elif msg_type == "error":
                stderr_parts.append(re.sub(r"\x1b\[[0-9;]*m", "", "".join(content.get("traceback", []))))
            elif msg_type in {"execute_result", "display_data"}:
                text = content.get("data", {}).get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else f"{text}\n")
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break
        stdout, stderr = "".join(stdout_parts), "".join(stderr_parts)
        if stderr:
            return f"{stdout.rstrip()}\n{stderr}" if stdout else stderr
        return stdout if stdout.strip() else "[WARN] No output. Use print() to see results."

    def reset(self):
        self.execute("%reset -f\nimport math\nimport numpy\nimport sympy\nimport itertools\nimport collections\nimport mpmath\nmpmath.mp.dps = 64\n")

    def close(self):
        with contextlib.suppress(Exception):
            if self._client:
                self._client.stop_channels()
        if self._owns_kernel and self._km:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)

    def __del__(self):
        self.close()


# ============================================================
# Template + Tool
# ============================================================

class AIMO3Template:
    def apply_chat_template(self, system_prompt, user_prompt, tool_config):
        system_content = (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
            .with_tools(tool_config)
        )
        return [
            Message.from_role_and_content(Role.SYSTEM, system_content),
            Message.from_role_and_content(Role.USER, user_prompt),
        ]


class AIMO3Tool:
    def __init__(self, jupyter_timeout, tool_prompt, sandbox=None):
        self._timeout = jupyter_timeout
        self._tool_prompt = tool_prompt
        self._sandbox = sandbox
        self._lock = threading.Lock()

    @property
    def tool_config(self):
        return ToolNamespaceConfig(name="python", description=self._tool_prompt, tools=[])

    def _ensure_last_print(self, code):
        lines = code.strip().split("\n")
        if not lines:
            return code
        last = lines[-1].strip()
        if not last or "print" in last or "import" in last or last.startswith("#"):
            return code
        lines[-1] = f"print({last})"
        return "\n".join(lines)

    def process_sync_plus(self, message):
        raw = message.content[0].text
        code = self._ensure_last_print(raw)
        with self._lock:
            try:
                output = self._sandbox.execute(code)
            except TimeoutError as exc:
                output = f"[ERROR] {exc}"
        resp = Message(
            author=Author(role=Role.TOOL, name="python"),
            content=[TextContent(text=output)],
        ).with_recipient("assistant")
        return [resp]


# ============================================================
# Main Solver
# ============================================================

class AIMO3Solver44:
    """
    Adapted from the 44/50 scoring notebook (parthenos).
    Connects to an EXISTING vLLM server (launched by Fix5Runtime).
    """

    def __init__(self, cfg=None, port=8000):
        self.cfg = cfg or CFG44()
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self.template = AIMO3Template()
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
        self.client = OpenAI(base_url=self.base_url, api_key="EMPTY", timeout=self.cfg.session_timeout)

        # Wait for existing vLLM server
        print("Waiting for vLLM server...")
        for _ in range(self.cfg.server_timeout):
            try:
                self.client.models.list()
                print("Server connected.")
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("Failed to connect to vLLM server.")

        # Initialize Jupyter sandbox pool
        print(f"Initializing {self.cfg.workers} Jupyter kernels...")
        self.sandbox_pool = queue.Queue()
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as ex:
            futs = [ex.submit(lambda: AIMO3Sandbox(timeout=self.cfg.jupyter_timeout)) for _ in range(self.cfg.workers)]
            for f in as_completed(futs):
                self.sandbox_pool.put(f.result())
        print(f"Kernels ready.")

        self.notebook_start_time = time.time()
        self.problems_remaining = 50

    def _scan_for_answer(self, text):
        for pat in [r"\\boxed\s*\{\s*([0-9,]+)\s*\}", r"final\s+answer\s+is\s*([0-9,]+)"]:
            matches = re.findall(pat, text, re.IGNORECASE)
            if matches:
                try:
                    val = int(matches[-1].replace(",", ""))
                    if 0 <= val <= 99999:
                        return val
                except ValueError:
                    pass
        return None

    def _compute_mean_entropy(self, logprobs_buffer):
        if not logprobs_buffer:
            return float("inf")
        total, count = 0.0, 0
        for top_lp in logprobs_buffer:
            if not isinstance(top_lp, dict):
                continue
            ent = 0.0
            for lp in top_lp.values():
                prob = math.exp(lp)
                if prob > 0:
                    ent -= prob * math.log2(prob)
            total += ent
            count += 1
        return total / count if count else float("inf")

    def _process_attempt(self, problem, system_prompt, attempt_index, stop_event, deadline):
        if stop_event.is_set() or time.time() > deadline:
            return {"Attempt": attempt_index + 1, "Answer": None, "Entropy": float("inf")}

        sandbox = None
        python_calls = 0
        final_answer = None
        logprobs_buffer = []
        attempt_seed = int(math.pow(self.cfg.seed + attempt_index, 2))

        try:
            sandbox = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)
            tool = AIMO3Tool(self.cfg.jupyter_timeout, self.cfg.tool_prompt, sandbox=sandbox)
            messages = self.template.apply_chat_template(system_prompt, problem, tool.tool_config)
            conversation = Conversation.from_messages(messages)

            for _ in range(self.cfg.turns):
                if stop_event.is_set() or time.time() > deadline:
                    break

                prompt_ids = self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
                max_tokens = self.cfg.context_tokens - len(prompt_ids)
                if max_tokens < self.cfg.buffer_tokens:
                    break

                stream = self.client.completions.create(
                    model=self.cfg.served_model_name,
                    temperature=self.cfg.temperature,
                    logprobs=self.cfg.top_logprobs,
                    max_tokens=max_tokens,
                    prompt=prompt_ids,
                    seed=attempt_seed,
                    stream=True,
                    extra_body={
                        "min_p": self.cfg.min_p,
                        "stop_token_ids": self.stop_token_ids,
                        "return_token_ids": True,
                    },
                )

                try:
                    token_buffer = []
                    text_chunks = []
                    for chunk in stream:
                        if stop_event.is_set() or time.time() > deadline:
                            break
                        new_tokens = chunk.choices[0].token_ids
                        new_text = chunk.choices[0].text
                        if new_tokens:
                            token_buffer.extend(new_tokens)
                            text_chunks.append(new_text)
                            cl = chunk.choices[0].logprobs
                            if cl and cl.top_logprobs:
                                logprobs_buffer.extend(cl.top_logprobs)
                        if "}" in new_text:
                            search_text = "".join(text_chunks[-self.cfg.search_tokens:])
                            answer = self._scan_for_answer(search_text)
                            if answer is not None:
                                final_answer = answer
                                break
                finally:
                    stream.close()

                if final_answer is not None:
                    break
                if not token_buffer:
                    break

                new_messages = self.encoding.parse_messages_from_completion_tokens(token_buffer, Role.ASSISTANT)
                conversation.messages.extend(new_messages)
                last_msg = new_messages[-1]

                if last_msg.channel == "final":
                    final_answer = self._scan_for_answer(last_msg.content[0].text)
                    break

                if last_msg.recipient == "python":
                    python_calls += 1
                    tool_responses = tool.process_sync_plus(last_msg)
                    conversation.messages.extend(tool_responses)

        except Exception:
            pass
        finally:
            if sandbox:
                sandbox.reset()
                self.sandbox_pool.put(sandbox)

        return {
            "Attempt": attempt_index + 1,
            "Answer": final_answer,
            "Entropy": self._compute_mean_entropy(logprobs_buffer),
            "Python Calls": python_calls,
        }

    def _select_answer(self, results):
        answer_weights = defaultdict(float)
        answer_votes = defaultdict(int)
        for r in results:
            a, e = r["Answer"], r["Entropy"]
            if a is not None:
                answer_weights[a] += 1.0 / max(e, 1e-9)
                answer_votes[a] += 1
        if not answer_weights:
            return 0
        return max(answer_weights, key=lambda a: (answer_weights[a], answer_votes[a]))

    def solve_problem(self, problem):
        user_input = f"{problem} {self.cfg.preference_prompt}"
        elapsed_global = time.time() - self.notebook_start_time
        time_left = self.cfg.notebook_limit - elapsed_global
        problems_left_others = max(0, self.problems_remaining - 1)
        reserved_time = problems_left_others * self.cfg.base_problem_timeout
        budget = time_left - reserved_time
        budget = min(budget, self.cfg.high_problem_timeout)
        budget = max(budget, self.cfg.base_problem_timeout)
        deadline = time.time() + budget
        print(f"Budget: {budget:.0f}s | Problems left: {self.problems_remaining}")

        stop_event = threading.Event()
        detailed_results = []
        valid_answers = []

        executor = ThreadPoolExecutor(max_workers=self.cfg.workers)
        try:
            futures = [
                executor.submit(self._process_attempt, user_input, self.cfg.system_prompt, i, stop_event, deadline)
                for i in range(self.cfg.attempts)
            ]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    detailed_results.append(result)
                    if result["Answer"] is not None:
                        valid_answers.append(result["Answer"])
                    counts = Counter(valid_answers).most_common(1)
                    if counts and counts[0][1] >= self.cfg.early_stop:
                        stop_event.set()
                        for f in futures:
                            f.cancel()
                        break
                except Exception:
                    continue
        finally:
            stop_event.set()
            executor.shutdown(wait=True, cancel_futures=True)
            self.problems_remaining = max(0, self.problems_remaining - 1)

        if not valid_answers:
            return 0
        return self._select_answer(detailed_results)

    def predict(self, problem_id, problem_text):
        """Interface for notebook compatibility."""
        return self.solve_problem(problem_text)

    def __del__(self):
        if hasattr(self, "sandbox_pool"):
            while not self.sandbox_pool.empty():
                try:
                    self.sandbox_pool.get_nowait().close()
                except Exception:
                    pass
