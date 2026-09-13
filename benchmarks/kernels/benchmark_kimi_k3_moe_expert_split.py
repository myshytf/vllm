# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native QSRT prefill expert-split identity and isolated latency gate."""

import argparse
import dataclasses
import hashlib
import json
import statistics
from pathlib import Path

import torch


def digest(x):
    return hashlib.sha256(
        x.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", default="1152,2304")
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    from b12x.moe import fused_moe
    from b12x.moe._shared.qsrt_sharding import plan_qsrt_tp9_rank

    from vllm.model_executor.layers.quantization.kquant_qsrt_atoms_v2 import (
        open_qsrt_atom_v2_extent,
        read_qsrt_atom_v2_layer_metadata,
    )

    records = []
    metadata = read_qsrt_atom_v2_layer_metadata(
        Path("/model") / f"qsrt-layer-{args.layer:05d}.safetensors", layer=args.layer
    )
    sizes = list(map(int, args.rows.split(",")))
    for width in (384, 256):
        rank = next(
            r
            for r in range(9)
            if plan_qsrt_tp9_rank(args.layer, r).intermediate_channels == width
        )
        weight_plan = fused_moe.plan_weights(
            quant_modes="w4a16",
            source_format="qsrt_sqg_e4m3",
            activation="situ",
            params_dtype=torch.bfloat16,
            num_experts=896,
            hidden_size=3584,
            intermediate_size=width,
            w13_layout="w13",
            trellis_bits=2,
            trellis_tile_config=(128, 128, 128, 128),
            qsrt_storage_format="qsrt_atoms_v2",
            qsrt_profile=metadata.profile,
        )
        with open_qsrt_atom_v2_extent(
            metadata, shard_count=9, shard_index=rank, device=None
        ) as (first, atoms):
            weights = fused_moe.prepare_weights(
                plan=weight_plan,
                params_dtype=torch.bfloat16,
                qsrt_atom_payload=atoms,
                qsrt_first_atom_slot=first,
                qsrt_layer_index=args.layer,
                gate_suh=metadata.gate_suh.unsqueeze(0).cuda(),
                up_suh=metadata.up_suh.unsqueeze(0).cuda(),
                down_svh=metadata.down_svh.unsqueeze(0).cuda(),
                qsrt_rotation_draws=metadata.rotation_draws,
            )
        plan = fused_moe.plan(
            fused_moe.Caps(
                max_tokens=max(sizes),
                num_topk=16,
                device=0,
                weight_plan=weights.plan,
                quant_mode="w4a16",
                route_num_experts=896,
                w4a16_block_size_m=48,
            )
        )
        spec = plan.scratch_specs()[0]
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        output = torch.empty((max(sizes), 3584), dtype=torch.bfloat16, device="cuda")
        mapping = torch.arange(896, dtype=torch.int32, device="cuda")
        first_map = torch.where(mapping < 448, mapping, -1)
        second_map = torch.where(mapping >= 448, mapping, -1)
        for rows in sizes:
            torch.manual_seed(20260914 + width + rows)
            x = (torch.randn(rows, 3584, device="cuda") * 0.1).to(torch.bfloat16)
            ids = torch.stack(
                [torch.randperm(896, device="cuda")[:16] for _ in range(rows)]
            ).to(torch.int32)
            routing = torch.softmax(torch.randn(rows, 16, device="cuda"), dim=-1)
            binding = fused_moe.bind(
                plan,
                scratch=scratch,
                a=x,
                experts=weights,
                topk_weights=routing,
                topk_ids=ids,
                route_expert_map=mapping,
                output=output[:rows],
            )
            retained_fc2 = torch.empty(
                rows * 16 * 3584,
                dtype=binding.intermediate_cache13.dtype,
                device="cuda",
            )
            first = dataclasses.replace(
                binding,
                route_expert_map=first_map,
                output_expert_map=mapping,
                skip_topk_sum=True,
                retained_fc2_output=retained_fc2,
            )
            second = dataclasses.replace(
                binding,
                route_expert_map=second_map,
                output_expert_map=mapping,
                zero_fc2_output_override=False,
                retained_fc2_output=retained_fc2,
            )
            expected = fused_moe.run(binding=binding).clone()
            fused_moe.run(binding=first)
            actual = fused_moe.run(binding=second)
            torch.accelerator.synchronize()
            exact = torch.equal(expected.view(torch.int16), actual.view(torch.int16))
            max_error = (expected.float() - actual.float()).abs().max().item()
            row = {
                "rows": rows,
                "width": width,
                "rank": rank,
                "exact": exact,
                "max_abs_error": max_error,
                "reference_sha256": digest(expected),
                "candidate_sha256": digest(actual),
            }
            if not exact:
                records.append(row)
                args.out.write_text(
                    json.dumps(
                        {"status": "research-only", "records": records}, indent=2
                    )
                )
                raise AssertionError(json.dumps(row))
            graphs = []
            for arm in range(2):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    if arm:
                        fused_moe.run(binding=first)
                        fused_moe.run(binding=second)
                    else:
                        fused_moe.run(binding=binding)
                graphs.append(graph)
            samples = [[], []]
            for iteration in range(0 if args.correctness_only else 30):
                for arm in (0, 1) if iteration % 2 == 0 else (1, 0):
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                    graphs[arm].replay()
                    end.record()
                    end.synchronize()
                    if iteration >= 6:
                        samples[arm].append(begin.elapsed_time(end))
            for mutation in range(3):
                x.normal_(std=0.1)
                ids.copy_(
                    torch.stack(
                        [torch.randperm(896, device="cuda")[:16] for _ in range(rows)]
                    )
                )
                routing.copy_(torch.softmax(torch.randn_like(routing), dim=-1))
                graphs[0].replay()
                expected.copy_(output[:rows])
                output.fill_(float("nan"))
                graphs[1].replay()
                assert torch.equal(
                    expected.view(torch.int16), output[:rows].view(torch.int16)
                )
                assert torch.isfinite(output[:rows]).all()
            row.update(
                {
                    "mutated_graph_exact": True,
                    "baseline_ms": statistics.median(samples[0])
                    if samples[0]
                    else None,
                    "candidate_ms": statistics.median(samples[1])
                    if samples[1]
                    else None,
                    "raw_ms": samples,
                }
            )
            records.append(row)
            print(
                json.dumps({k: v for k, v in row.items() if k != "raw_ms"}), flush=True
            )
            args.out.write_text(
                json.dumps({"status": "research-only", "records": records}, indent=2)
            )
            del graph, graphs, first, second, binding, expected, actual, x, ids, routing
        del weights, plan, scratch, output, mapping, first_map, second_map
        torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
