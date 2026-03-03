#!/usr/bin/env python3
"""Minimal throughput test: send N real requests across 8 SGLang endpoints, no pipeline overhead.
Measures: image encoding time, HTTP round-trip time, total throughput.

Usage:
  python scripts/tasks/bench_raw_throughput.py --input /ov2/dataset_jsonl/openbee/MAVIS_Function.jsonl --n 100
"""
import argparse
import asyncio
import base64
import io
import json
import os
import time
from pathlib import Path

import httpx

ENDPOINTS = [f"http://127.0.0.1:{p}/v1/chat/completions" for p in range(10025, 10033)]
MODEL = "Qwen/Qwen3-VL-32B-Instruct"
SYSTEM = "Judge whether image and question are relevant. Reply JSON: {relevance, necessity, reason}."


def load_items(path: str, n: int):
    items = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            # Extract question from messages
            q = rec.get("question") or ""
            if not q and isinstance(rec.get("messages"), list):
                for m in rec["messages"]:
                    if isinstance(m, dict) and m.get("role") == "user":
                        content = str(m.get("content", ""))
                        content = content.replace("<image>", "").strip()
                        if content:
                            q = content
                            break
            # Extract image path
            img = rec.get("image") or ""
            if not img and isinstance(rec.get("images"), list) and rec["images"]:
                img = str(rec["images"][0])
            if not img and isinstance(rec.get("images_source"), list) and rec["images_source"]:
                img = str(rec["images_source"][0])
            if q and img and os.path.isfile(img):
                items.append({"question": q[:2000], "image_path": img})
            if len(items) >= n:
                break
    return items


def encode_image(path: str, max_edge: int = 1024) -> str:
    from PIL import Image
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_edge:
        ratio = max_edge / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def build_payload(item: dict, encoded_b64: str) -> dict:
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": f"Question:\n{item['question']}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_b64}", "detail": "low"}},
            ]},
        ],
        "max_tokens": 512,
        "temperature": 0.0,
    }


async def send_one(client: httpx.AsyncClient, url: str, payload: dict, idx: int, stats: dict):
    t0 = time.monotonic()
    try:
        resp = await client.post(url, json=payload, timeout=90)
        elapsed = time.monotonic() - t0
        stats["http_times"].append(elapsed)
        if resp.status_code == 200:
            stats["ok"] += 1
        else:
            stats["fail"] += 1
            print(f"  [{idx}] HTTP {resp.status_code} from {url} ({elapsed:.2f}s)")
    except Exception as e:
        elapsed = time.monotonic() - t0
        stats["fail"] += 1
        stats["http_times"].append(elapsed)
        print(f"  [{idx}] Error: {e} ({elapsed:.2f}s)")


async def run_concurrent(items: list[dict], concurrency: int):
    stats = {"ok": 0, "fail": 0, "http_times": [], "encode_times": []}
    n = len(items)

    # Pre-encode images (measure encoding time)
    print(f"Encoding {n} images...")
    t_enc_start = time.monotonic()
    payloads = []
    for i, item in enumerate(items):
        t0 = time.monotonic()
        b64 = encode_image(item["image_path"])
        stats["encode_times"].append(time.monotonic() - t0)
        url = ENDPOINTS[i % len(ENDPOINTS)]
        payloads.append((url, build_payload(item, b64), i))
    t_enc_total = time.monotonic() - t_enc_start
    avg_enc = sum(stats["encode_times"]) / len(stats["encode_times"]) if stats["encode_times"] else 0
    print(f"  Encoding done: {t_enc_total:.2f}s total, {avg_enc*1000:.1f}ms avg per image")

    # Send all requests concurrently with semaphore
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        async def task(url, payload, idx):
            async with sem:
                await send_one(client, url, payload, idx, stats)

        print(f"Sending {n} requests (concurrency={concurrency}) across {len(ENDPOINTS)} endpoints...")
        t_http_start = time.monotonic()
        await asyncio.gather(*[task(url, pl, idx) for url, pl, idx in payloads])
        t_http_total = time.monotonic() - t_http_start

    avg_http = sum(stats["http_times"]) / len(stats["http_times"]) if stats["http_times"] else 0
    print(f"\n=== Results ===")
    print(f"  Items:       {n}")
    print(f"  Concurrency: {concurrency}")
    print(f"  Endpoints:   {len(ENDPOINTS)}")
    print(f"  OK/Fail:     {stats['ok']}/{stats['fail']}")
    print(f"  Encode:      {t_enc_total:.2f}s total, {avg_enc*1000:.1f}ms avg/img")
    print(f"  HTTP:        {t_http_total:.2f}s total, {avg_http:.2f}s avg/req")
    print(f"  Throughput:  {n / t_http_total:.2f} rec/s (HTTP only)")
    print(f"  Throughput:  {n / (t_enc_total + t_http_total):.2f} rec/s (encode + HTTP)")
    p50 = sorted(stats["http_times"])[len(stats["http_times"]) // 2] if stats["http_times"] else 0
    p99 = sorted(stats["http_times"])[int(len(stats["http_times"]) * 0.99)] if stats["http_times"] else 0
    print(f"  Latency:     p50={p50:.2f}s p99={p99:.2f}s")
    ep_dist = {}
    for url, _, _ in payloads:
        port = url.split(":")[2].split("/")[0]
        ep_dist[port] = ep_dist.get(port, 0) + 1
    print(f"  Distribution: {ep_dist}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=64)
    args = parser.parse_args()

    items = load_items(args.input, args.n)
    print(f"Loaded {len(items)} items from {args.input}")
    asyncio.run(run_concurrent(items, args.concurrency))


if __name__ == "__main__":
    main()
