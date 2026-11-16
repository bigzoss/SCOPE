"""Evaluate a frozen HotpotQA adapter and AgeMem-noRL on paired shopping tasks."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scope.environments import ShopEnvironment, parse_action
from scope.io import atomic_json, file_sha
from scope.launcher import own_service
from trinity.common.workflows.memory_context.eval_hotpotQA import AgeMemHotpotWorkflowEvaluation
from trinity.common.workflows.memory_context.utils import parse_tool_calls
from trinity.common.workflows.workflow import Task


SHOP_ACTION_LIMIT = 30
MODEL_DECISION_LIMIT = 60
PROMPT_TOKEN_LIMIT = 8192
RESPONSE_TOKEN_LIMIT = 512
# The episode token budget should not cut off the configured number of turns.
TOTAL_TOKEN_LIMIT = MODEL_DECISION_LIMIT * (PROMPT_TOKEN_LIMIT + RESPONSE_TOKEN_LIMIT)


RULES = """你是购物助手。依据购物需求、历史和当前页面选择商品。
每轮可调用 AgeMem 记忆工具，或者输出一条购物操作，两者只能选其一。
购物操作格式是 Action: search[关键词] 或 Action: click[页面按钮]。
购买前确认商品和规格，最后执行 Action: click[Buy Now]。不要输出问答用的 <answer> 标签。
记忆工具用 <tool_call> 包裹 JSON 数组，每项包含 name 和 arguments。"""


class ShopAgeMem(AgeMemHotpotWorkflowEvaluation):
    def start(self, goal):
        self.reset_per_run()
        self.current_stage = 3
        self.question = goal
        rules = RULES + "\n可用的 AgeMem 工具：\n" + json.dumps(self.tool_schema, ensure_ascii=False)
        self._append_context("system", rules)

    def prompt(self, goal, observation):
        # The task goal and latest public page remain available after context edits.
        return self.context_messages + [
            {"role": "user", "content": f"购物需求：{goal}\n当前页面：{observation}"}
        ]


def episode(task_id, env, generate):
    agent = ShopAgeMem(Task(repeat_times=1), model=None)
    trace, calls, actions = [], 0, 0
    termination = "model_decision_budget"
    input_tokens, output_tokens = 0, 0
    try:
        goal = observation = env.reset(task_id)
        agent.start(goal)
        for _ in range(MODEL_DECISION_LIMIT):
            generated = generate(
                agent.prompt(goal, observation), TOTAL_TOKEN_LIMIT - input_tokens - output_tokens
            )
            if generated is None:
                termination = "token_budget"
                break
            text, in_count, out_count = generated
            input_tokens += in_count
            output_tokens += out_count
            raw = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
            tools = parse_tool_calls(raw)
            action = parse_action(raw)
            if action and not tools:
                agent._append_context("user", observation)
            agent._append_context("assistant", text)
            if tools and action is None:
                agent._apply_tools(tools)
                calls += len(tools)
                trace.append({"memory_tools": tools})
            elif action and not tools:
                observation, done, info = env.step_command(action)
                actions += 1
                trace.append({"action": action, "invalid": info["is_error"]})
                if done or actions == SHOP_ACTION_LIMIT:
                    termination = "purchase" if done else "shop_action_budget"
                    break
            else:
                agent._append_context("tool", "格式错误。请输出一条购物操作或一组 AgeMem 工具调用。")
                trace.append({"invalid_output": raw})
        if agent.chat_client.failed:
            raise RuntimeError("Shopping task has an incomplete auxiliary API call")
        return dict(
            task_id=task_id, **env.metrics, memory_tool_calls=calls, actions=actions,
            model_decisions=len(trace), termination_reason=termination,
            input_tokens=input_tokens, output_tokens=output_tokens, trace=trace,
        )
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--port", type=int, default=5149)
    args = parser.parse_args()
    spec = json.loads((args.run / "protocol-ustc.json").read_text())
    shop = spec["shopsimulator"]
    assert file_sha(shop["manifest"]) == shop["manifest_sha256"]
    adapter = args.adapter.resolve()
    assert (adapter / "adapter_config.json").is_file()
    assert (adapter / "adapter_model.safetensors").is_file()
    project = Path(__file__).resolve().parents[1]
    cfg = json.loads((project / "configs/shopsim.json").read_text())
    cfg["environment"].update(manifest=shop["manifest"], base_url=f"http://127.0.0.1:{args.port}")
    output = args.output or args.run / "shop_30_60"
    task_ids = shop["test_ids"]
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("Shop evaluation requires a nonempty task list with unique IDs")
    receipt = {
        "adapter": str(adapter),
        "adapter_sha256": file_sha(adapter / "adapter_model.safetensors"),
        "source_sha256": file_sha(__file__),
        "protocol_sha256": file_sha(args.run / "protocol-ustc.json"),
        "task_ids": task_ids,
        "manifest_sha256": shop["manifest_sha256"],
        "services": spec["services"],
        "budgets": {
            "model_decisions": MODEL_DECISION_LIMIT,
            "shop_actions": SHOP_ACTION_LIMIT,
            "prompt": PROMPT_TOKEN_LIMIT,
            "response": RESPONSE_TOKEN_LIMIT,
            "total": TOTAL_TOKEN_LIMIT,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    receipt_file = output / "receipt.json"
    if receipt_file.exists() and json.loads(receipt_file.read_text()) != receipt:
        raise ValueError("Shop evaluation receipt differs")
    atomic_json(receipt_file, receipt)
    tokenizer = AutoTokenizer.from_pretrained(spec["base_model"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        spec["base_model"], torch_dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True
    )
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False).eval()

    def generate(messages, remaining):
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False,
            tokenize=True, return_tensors="pt",
        ).to("cuda")
        in_count = ids.shape[-1]
        budget = min(RESPONSE_TOKEN_LIMIT, remaining - in_count)
        if in_count > PROMPT_TOKEN_LIMIT or budget <= 0:
            return None
        with torch.inference_mode():
            response = model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=budget,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )[0, in_count:]
        tokens = response.tolist()
        visible = tokens[:-1] if tokens and tokens[-1] in tokenizer.all_special_ids else tokens
        return tokenizer.decode(visible, skip_special_tokens=False), in_count, len(tokens)

    rows = {}
    with own_service(cfg, output, enabled=True):
        env = ShopEnvironment(cfg["environment"]["base_url"], shop["test_ids"], shop["manifest_sha256"])
        env.health()
        for name in ("trained", "noRL"):
            folder = output / name
            folder.mkdir(exist_ok=True)
            rows[name] = []
            for index, task_id in enumerate(shop["test_ids"]):
                path = folder / f"{index:04d}.json"
                if not path.exists():
                    if name == "noRL":
                        with model.disable_adapter():
                            result = episode(task_id, env, generate)
                    else:
                        result = episode(task_id, env, generate)
                    atomic_json(path, result)
                result = json.loads(path.read_text())
                if (result["task_id"] != task_id or
                    result["model_decisions"] != len(result["trace"]) or
                    result["actions"] != sum("action" in event for event in result["trace"]) or
                    not 0 <= result["actions"] <= SHOP_ACTION_LIMIT or
                    not 0 <= result["model_decisions"] <= MODEL_DECISION_LIMIT or
                    bool(result["purchased"]) != (result["termination_reason"] == "purchase")):
                    raise ValueError(f"Invalid shopping episode record: {path}")
                rows[name].append(result)
                print(name, index + 1, len(shop["test_ids"]), flush=True)
    scores = ("success", "r_strict", "r_loose", "purchased", "actions", "model_decisions",
              "input_tokens", "output_tokens", "memory_tool_calls")
    summary = {}
    for name, result in rows.items():
        summary[name] = {score: sum(row[score] for row in result) / len(result) for score in scores}
        summary[name]["success_count"] = sum(int(row["success"]) for row in result)
        summary[name]["termination_reasons"] = dict(Counter(row["termination_reason"] for row in result))
    summary["paired_success"] = {
        "trained_only": sum(a["success"] > b["success"] for a, b in zip(rows["trained"], rows["noRL"])),
        "noRL_only": sum(a["success"] < b["success"] for a, b in zip(rows["trained"], rows["noRL"])),
        "n": len(shop["test_ids"]),
    }
    atomic_json(output / "summary.json", {"complete": True, "budgets": receipt["budgets"], **summary})


if __name__ == "__main__":
    main()
