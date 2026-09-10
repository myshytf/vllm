# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure native BF16 versus packed-record DCP publishing on two CUDA peers.

Both paths start at the same paged FP8 cache and finish with equal BF16 planes.
The packed path sends the original 656-byte records and expands after receipt.
Run on explicitly selected idle devices; kernel timings are not serving claims.
"""

import argparse
import json
import os
import socket
import statistics
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, port, args):
    from vllm import _custom_ops as ops
    from vllm.v1.attention.ops.dcp_utils import (
        DirectDCPKVGatherWorkspace,
        MLADCPKVGather,
    )
    from vllm.v1.attention.ops.kimi_packed_kv_transport import gather_packed_records

    with torch.accelerator.device_index(rank):
        device = torch.device("cuda", rank)
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=2,
            device_id=device,
            timeout=timedelta(seconds=180),
        )
        local_rows = 3072
        total_rows = 2 * local_rows
        cache = torch.randint(0, 256, (2, 1536, 656), dtype=torch.uint8, device=device)
        nope = cache[..., :512]
        nope.copy_(torch.where((nope & 127) == 127, nope - 1, nope))
        cache[..., 512:528].view(torch.float32).fill_(0.25 * (rank + 1))
        cache[..., 528:].view(torch.bfloat16).fill_(rank + 1)
        table = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
        zero = torch.zeros(1, dtype=torch.int32, device=device)
        token_map = torch.zeros(local_rows, dtype=torch.int32, device=device)
        raw = torch.empty((local_rows, 656), dtype=torch.uint8, device=device)
        expanded = torch.empty((local_rows, 576), dtype=torch.bfloat16, device=device)
        local_c = torch.empty((local_rows, 512), dtype=torch.bfloat16, device=device)
        local_r = torch.empty((local_rows, 64), dtype=torch.bfloat16, device=device)
        runs = torch.tensor(
            [[0, rank * local_rows, local_rows]], dtype=torch.int64, device="cpu"
        )
        arms = {}
        reference = None
        for name, packed in (("bf16_wire", False), ("packed_wire", True)):
            workspace = DirectDCPKVGatherWorkspace(
                dist.group.WORLD,
                device,
                total_rows,
                656 if packed else 576,
                528 if packed else 512,
                torch.float8_e4m3fn if packed else torch.bfloat16,
                1,
            )
            manager = MLADCPKVGather(SimpleNamespace(world_size=2), device, 1)
            manager._direct_kv_gather_workspace = workspace

            def run(packed=packed, manager=manager, workspace=workspace):
                if packed:
                    gather_packed_records(cache, raw, table, token_map, zero, zero)
                    c, r = manager.packed_dma_planes(local_rows)
                    c.copy_(raw[:, :528].view(torch.float8_e4m3fn))
                    r.copy_(raw[:, 528:].view(torch.float8_e4m3fn))
                else:
                    ops.cp_gather_and_upconvert_fp8_kv_cache(
                        cache, expanded, table, zero, 1, seq_starts=zero
                    )
                    local_c.copy_(expanded[:, :512])
                    local_r.copy_(expanded[:, 512:])
                    c, r = local_c, local_r
                received = workspace.gather_dma(c, r, runs, total_rows, 0)
                return manager.unpack_context_planes(*received) if packed else received

            output = run()
            torch.accelerator.synchronize()
            if reference is None:
                reference = tuple(t.cpu() for t in output)
            else:
                assert all(
                    torch.equal(t.cpu().view(torch.uint16), ref.view(torch.uint16))
                    for t, ref in zip(output, reference, strict=True)
                )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = run()
            graph.replay()
            torch.accelerator.synchronize()
            assert all(
                torch.equal(t.cpu().view(torch.uint16), ref.view(torch.uint16))
                for t, ref in zip(output, reference, strict=True)
            )
            arms[name] = dict(
                graph=graph, output=output, workspace=workspace, manager=manager
            )
        samples = {name: [] for name in arms}
        for arm in arms.values():
            for _ in range(10):
                arm["graph"].replay()
        torch.accelerator.synchronize()
        for repeat in range(9):
            for name in list(arms) if repeat % 2 == 0 else list(reversed(arms)):
                dist.barrier()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(30):
                    arms[name]["graph"].replay()
                end.record()
                end.synchronize()
                elapsed = torch.tensor(
                    start.elapsed_time(end) * 1000 / 30, device=device
                )
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                samples[name].append(elapsed.item())
                assert all(
                    torch.equal(t.cpu().view(torch.uint16), ref.view(torch.uint16))
                    for t, ref in zip(arms[name]["output"], reference, strict=True)
                )
        if rank == 0:
            medians = {k: statistics.median(v) for k, v in samples.items()}
            report = dict(
                status="qualified two-peer kernel screen",
                local_rows=local_rows,
                bytes_per_record={"bf16_wire": 1152, "packed_wire": 656},
                medians_us=medians,
                samples_us=samples,
                packed_over_bf16=medians["packed_wire"] / medians["bf16_wire"],
            )
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
        dist.barrier()
        del arms, workspace, manager, graph, output, run
        torch.accelerator.synchronize()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.environ["VLLM_DCP_KV_GATHER_SLOTS"] = "3"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(port, args), nprocs=2, join=True)


if __name__ == "__main__":
    main()
