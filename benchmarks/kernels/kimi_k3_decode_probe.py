# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker extension for isolated, reversible Kimi L2-prefetch experiments.

Load explicitly with --worker-extension-cls. Call through collective_rpc only
after closing ingress and draining requests. The extension edits graph-owned
prefetch descriptors, never weights, activations, attention state or arithmetic.
Changing descriptor contents avoids recompiling graphs between comparison arms.
Restore ``all`` and the original persisting-L2 limit before reopening ingress.
"""

from __future__ import annotations

import torch


class KimiDecodeProbeWorker:
    def _k3_prefetch_plans(self):
        if hasattr(self, "_k3_probe_plans"):
            return self._k3_probe_plans
        from vllm.models.kimi_k3.nvidia.l2_prefetch import L2PrefetchPlan

        plans = []
        seen = set()
        for name, module in self.get_model().named_modules():
            for hook_name in (
                "_l2_prefetch_hook",
                "_l2_prefetch_pre_reduce_hook",
            ):
                hook = getattr(module, hook_name, None)
                defaults = getattr(hook, "__defaults__", None) or ()
                for plan in defaults:
                    if not isinstance(plan, L2PrefetchPlan) or id(plan) in seen:
                        continue
                    seen.add(id(plan))
                    if plan.segs is None:
                        continue
                    original = plan.segs.detach().cpu().clone()
                    assert original.numel() == 2 * plan.nseg
                    assert int(original[1::2].sum()) == plan.total_bytes
                    window = "A" if hook_name == "_l2_prefetch_hook" else "BC"
                    plans.append((name, window, plan, original))
        if not plans:
            raise RuntimeError("No initialized target-model prefetch plans")
        self._k3_probe_plans = plans
        return plans

    def k3_decode_probe_status(self):
        """Describe the actual descriptors and graph batch sizes on this rank."""
        torch.accelerator.synchronize()
        rows = []
        for name, window, plan, original in self._k3_prefetch_plans():
            current = plan.segs.cpu()
            rows.append(
                {
                    "module": name,
                    "window": window,
                    "segments": plan.names,
                    "original_bytes": int(original[1::2].sum()),
                    "active_bytes": int(current[1::2].sum()),
                    "addresses_equal": torch.equal(current[::2], original[::2]),
                }
            )
        manager = self.model_runner.cudagraph_manager
        return {
            "rank": self.rank,
            "plans": rows,
            "graphs": [str(key) for key in manager.graphs],
            "policy": getattr(self, "_k3_probe_policy", "all"),
        }

    @torch.inference_mode()
    def k3_set_prefetch_policy(self, policy: str):
        """Select original/all, no reads, window A only, or windows B/C only."""
        if policy not in ("all", "off", "a_only", "bc_only"):
            raise ValueError("policy must be all, off, a_only or bc_only")
        torch.accelerator.synchronize()
        active_bytes = 0
        count = 0
        for _, window, plan, original in self._k3_prefetch_plans():
            enabled = (
                policy == "all"
                or (policy == "a_only" and window == "A")
                or (policy == "bc_only" and window == "BC")
            )
            values = original.clone()
            if not enabled:
                values[1::2].zero_()
            address = plan.segs.data_ptr()
            plan.segs.copy_(values)
            assert plan.segs.data_ptr() == address
            assert torch.equal(plan.segs.cpu(), values)
            active_bytes += int(values[1::2].sum())
            count += 1
        torch.accelerator.synchronize()
        self._k3_probe_policy = policy
        return {
            "rank": self.rank,
            "policy": policy,
            "plans": count,
            "active_bytes": active_bytes,
        }

    def k3_set_persisting_l2(self, requested: str):
        """Set and read back cache residency, including restoration to zero."""
        from cuda.bindings import driver as cu

        from vllm.models.kimi_k3.nvidia.l2_prefetch import persisting_l2_request

        if requested not in ("0", "32", "64", "max", "restore"):
            raise ValueError("requested must be 0, 32, 64, max or restore")

        def checked(result):
            if result[0] != cu.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"CUDA cache-policy operation failed: {result[0]}")
            return result[1] if len(result) > 1 else None

        torch.accelerator.synchronize()
        dev = checked(cu.cuDeviceGet(self.device.index))
        ctx = checked(cu.cuDevicePrimaryCtxRetain(dev))
        checked(cu.cuCtxPushCurrent(ctx))
        try:
            limit = cu.CUlimit.CU_LIMIT_PERSISTING_L2_CACHE_SIZE
            before = int(checked(cu.cuCtxGetLimit(limit)))
            if not hasattr(self, "_k3_probe_original_l2"):
                self._k3_probe_original_l2 = before
            maximum = checked(
                cu.cuDeviceGetAttribute(
                    cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE,
                    dev,
                )
            )
            wanted = (
                self._k3_probe_original_l2
                if requested == "restore"
                else persisting_l2_request(requested, maximum)
            )
            checked(cu.cuCtxSetLimit(limit, wanted))
            after = int(checked(cu.cuCtxGetLimit(limit)))
        finally:
            checked(cu.cuCtxPopCurrent())
            checked(cu.cuDevicePrimaryCtxRelease(dev))
        return {
            "rank": self.rank,
            "before": before,
            "requested": wanted,
            "after": after,
        }
