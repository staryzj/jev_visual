"""Lightweight, non-blocking result window for the live Visual-JEV demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageTk


def show(payload_path: Path) -> None:
    import tkinter as tk
    from tkinter import ttk

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    case = payload["case"]
    result = payload["result"]

    root = tk.Tk()
    root.title("Visual-JEV: Live Image to Decision")
    root.geometry("1420x820")
    root.minsize(1080, 680)
    root.lift()
    root.attributes("-topmost", True)
    root.after(600, lambda: root.attributes("-topmost", False))

    outer = ttk.Frame(root, padding=18)
    outer.pack(fill="both", expand=True)
    outer.columnconfigure(0, weight=4)
    outer.columnconfigure(1, weight=6)
    outer.rowconfigure(0, weight=1)

    left = ttk.Frame(outer)
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
    image = Image.open(case["image"]).convert("RGB")
    image.thumbnail((540, 570), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(image)
    image_label = ttk.Label(left, image=photo)
    image_label.image = photo
    image_label.pack(anchor="center", pady=(0, 12))
    ttk.Label(
        left,
        text=case["image"],
        wraplength=520,
        foreground="#555555",
    ).pack(anchor="w")

    right = ttk.Frame(outer)
    right.grid(row=0, column=1, sticky="nsew")
    ttk.Label(
        right,
        text="LIVE IMAGE → DECISION",
        font=("Arial", 22, "bold"),
    ).pack(anchor="w")
    ttk.Label(
        right,
        text=case["question"],
        font=("Arial", 16),
        wraplength=780,
    ).pack(anchor="w", pady=(12, 18))

    columns = ("candidate", "score", "probability", "result")
    table = ttk.Treeview(
        right,
        columns=columns,
        show="headings",
        height=max(4, len(case["candidates"])),
    )
    table.heading("candidate", text="Candidate")
    table.heading("score", text="Raw score")
    table.heading("probability", text="Softmax probability")
    table.heading("result", text="")
    table.column("candidate", width=330, anchor="w")
    table.column("score", width=120, anchor="e")
    table.column("probability", width=165, anchor="e")
    table.column("result", width=105, anchor="center")
    for item in result["candidates"]:
        top = item["index"] == result["prediction_index"]
        table.insert(
            "",
            "end",
            values=(
                item["text"],
                f"{item['raw_score']:.6f}",
                f"{item['probability'] * 100:.2f}%",
                "TOP-1" if top else "",
            ),
            tags=("top",) if top else (),
        )
    table.tag_configure("top", background="#dff5e1", foreground="#075e20")
    table.pack(fill="x")

    ttk.Label(
        right,
        text=f"Prediction: {result['prediction']}",
        font=("Arial", 18, "bold"),
        foreground="#075e20",
    ).pack(anchor="w", pady=(20, 8))
    history = result.get("history", {})
    if history.get("used"):
        turn_ids = ", ".join(str(value) for value in history["selected_turn_ids"])
        history_text = (
            f"Dialogue memory: used turn(s) {turn_ids} "
            f"· {history['history_prompt_tokens']} tokens"
        )
    else:
        history_text = f"Dialogue memory: not used · {history.get('reason', 'empty')}"
    ttk.Label(
        right,
        text=history_text,
        foreground="#315a83",
    ).pack(anchor="w", pady=(0, 8))
    if result.get("ground_truth") is not None:
        ttk.Label(
            right,
            text=(
                f"Ground truth: {result['ground_truth']}   ·   "
                f"{'CORRECT' if result['correct'] else 'INCORRECT'}"
            ),
            font=("Arial", 13),
        ).pack(anchor="w")

    latency = result["latency_ms"]
    timing_text = (
        f"Qwen visual  {latency['qwen_visual']:.3f} ms     "
        f"Qwen text  {latency['qwen_text']:.3f} ms\n"
        f"Adapter  {latency['alignment_adapter']:.3f} ms     "
        f"JEV  {latency['jev_decision']:.3f} ms\n"
        f"FULL IMAGE → DECISION  {latency['image_to_decision_e2e']:.3f} ms"
    )
    ttk.Separator(right).pack(fill="x", pady=18)
    ttk.Label(right, text=timing_text, font=("Consolas", 13)).pack(anchor="w")

    validation = result.get("cache_validation")
    if validation:
        ttk.Label(
            right,
            text=(
                "✓ Live pre-merger tokens match training cache "
                f"{tuple(validation['live_shape'])}; cache was not used for inference"
            ),
            foreground="#075e20",
        ).pack(anchor="w", pady=(18, 0))

    ttk.Label(
        right,
        text="结果窗口不会阻塞终端；可以直接在终端输入下一题。",
        foreground="#555555",
    ).pack(anchor="e", pady=(22, 4))
    ttk.Button(
        right,
        text="关闭结果窗口",
        command=root.destroy,
    ).pack(anchor="e")
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    args = parser.parse_args()
    try:
        show(args.payload)
    finally:
        args.payload.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
