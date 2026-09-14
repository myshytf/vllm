# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare shared-memory reservations using the deployed Marlin device binary.

Run on a dedicated idle GPU. Synthetic MXFP8 fixtures exercise the existing
format; this is component evidence, not a model-quality or serving speed claim.
"""

import argparse
import ctypes
import hashlib
import json
import statistics
from pathlib import Path

import torch
from cuda.bindings import runtime as cudart

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    prepare_mxfp8_layer_for_marlin,
)
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import current_stream, set_default_torch_dtype


def checked(result):
    if int(result[0]) != 0:
        raise RuntimeError(f"CUDA operation failed: {result[0]}")
    return result[1:]


def digest(tensor):
    raw = tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", default="1,4,8")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    # Establish vLLM's stream before producing any input consumed by the adapter.
    current_stream()
    torch.empty(1, device="cuda")
    free, total = checked(cudart.cudaMemGetInfo())
    if free < 2 * 2**30:
        raise RuntimeError("Dedicated probe requires 2 GiB free after context setup")
    (props,) = checked(cudart.cudaGetDeviceProperties(0))
    if (props.major, props.minor) != (12, 0):
        raise RuntimeError("This qualification targets SM120")
    sms, maximum = props.multiProcessorCount, props.sharedMemPerBlockOptin
    lib = ctypes.CDLL(str(args.library))
    launch = lib.marlin_probe_launch
    launch.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 10 + [ctypes.c_void_p]
    launch.restype = ctypes.c_int
    name_fn = lib.marlin_probe_name
    name_fn.argtypes = [ctypes.c_int] * 3 + [ctypes.POINTER(ctypes.c_char_p)]
    name_fn.restype = ctypes.c_int
    records = []
    for n, k, threads, n_blocks in ((1536, 7168, 128, 4), (7168, 768, 256, 8)):
        torch.manual_seed(20260914 + n)
        layer = torch.nn.Module()
        weight = (torch.randn(n, k, device="cuda") / 4).to(torch.float8_e4m3fn)
        scales = torch.randint(118, 132, (n, k // 32), dtype=torch.uint8, device="cuda")
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
        layer.output_size_per_partition = n
        layer.input_size_per_partition = k
        del weight, scales
        with set_default_torch_dtype(torch.bfloat16):
            prepare_mxfp8_layer_for_marlin(layer)
        before_hashes = [digest(layer.weight), digest(layer.weight_scale)]
        required = 4 * 128 * (16 * n_blocks) + 4 * 8 * 128 * 2 + 16 * (16 * n_blocks)
        kernel_name = ctypes.c_char_p()
        assert name_fn(threads, n_blocks, 8, ctypes.byref(kernel_name)) == 0
        assert kernel_name.value and b"Marlin" in kernel_name.value
        for m in map(int, args.rows.split(",")):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            outputs = [
                torch.empty(m, n, dtype=torch.bfloat16, device="cuda") for _ in range(2)
            ]
            temporary = torch.empty(sms * 16 * 256, dtype=torch.float32, device="cuda")
            locks = torch.zeros(sms, dtype=torch.int32, device="cuda")

            def run(
                arm,
                x=x,
                layer=layer,
                outputs=outputs,
                temporary=temporary,
                locks=locks,
                m=m,
                n=n,
                k=k,
                threads=threads,
                n_blocks=n_blocks,
                required=required,
            ):
                code = launch(
                    x.data_ptr(),
                    layer.weight.data_ptr(),
                    outputs[arm].data_ptr(),
                    temporary.data_ptr(),
                    layer.weight_scale.data_ptr(),
                    locks.data_ptr(),
                    m,
                    n,
                    k,
                    k,
                    threads,
                    n_blocks,
                    8,
                    sms,
                    maximum if arm == 0 else required,
                    maximum,
                    current_stream().cuda_stream,
                )
                if code:
                    raise RuntimeError(f"Native launch failed: {code}")

            def reference(x=x, layer=layer, m=m, n=n, k=k):
                return ops.marlin_gemm(
                    a=x,
                    c=None,
                    b_q_weight=layer.weight,
                    b_bias=None,
                    b_scales=layer.weight_scale,
                    a_scales=None,
                    global_scale=None,
                    b_zeros=None,
                    g_idx=None,
                    perm=None,
                    workspace=layer.workspace,
                    b_q_type=scalar_types.float8_e4m3fn,
                    size_m=m,
                    size_n=n,
                    size_k=k,
                    use_atomic_add=False,
                    use_fp32_reduce=True,
                )

            expected = reference()
            graphs = []
            for arm in range(2):
                temporary.fill_(float("nan"))
                outputs[arm].fill_(float("nan"))
                run(arm)
                torch.accelerator.synchronize()
                torch.testing.assert_close(outputs[arm], expected, rtol=0, atol=0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(20):
                        run(arm)
                graphs.append(graph)
            for mutation in range(3):
                x.copy_(torch.randn_like(x))
                expected = reference()
                for graph in graphs:
                    graph.replay()
                torch.accelerator.synchronize()
                for output in outputs:
                    assert torch.isfinite(output).all() and torch.count_nonzero(output)
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)
            samples = [[], []]
            for repeat in range(24):
                for arm in (0, 1) if repeat % 2 == 0 else (1, 0):
                    begin, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    begin.record()
                    graphs[arm].replay()
                    end.record()
                    end.synchronize()
                    if repeat >= 4:
                        samples[arm].append(begin.elapsed_time(end) * 1000 / 20)
            medians = list(map(statistics.median, samples))
            row = dict(
                m=m,
                n=n,
                k=k,
                threads=threads,
                grid=sms,
                original_shared_bytes=maximum,
                required_shared_bytes=required,
                original_us=medians[0],
                candidate_us=medians[1],
                speedup=medians[0] / medians[1],
                samples_us=samples,
                native_wrapper_and_graph_exact=True,
                output_sha256=digest(outputs[0]),
                kernel_name=kernel_name.value.decode(),
            )
            records.append(row)
            print(
                json.dumps({k: v for k, v in row.items() if k != "samples_us"}),
                flush=True,
            )
            (args.out / "results.json").write_text(json.dumps(records, indent=2))
            del graphs, graph, temporary, outputs, locks, x, expected, run, reference
        assert before_hashes == [digest(layer.weight), digest(layer.weight_scale)]
        del layer
    receipt = dict(
        status="research-only",
        torch=torch.__version__,
        cuda=torch.version.cuda,
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        device_binary="Deployed vLLM _C_stable_libtorch.abi3.so; not recompiled",
        initial_free_bytes=free,
        total_bytes=total,
        limitation=(
            "Synthetic existing-format fixtures and warm-weight component timing only."
        ),
    )
    (args.out / "manifest.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
