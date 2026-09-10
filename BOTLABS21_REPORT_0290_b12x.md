# GLM-5.3 (744B MoE, Int4/Int8 GPTQ) on vLLM 0.29.0 + b12x across 8× DGX Spark (GB10) — build & test report

**Status: DRAFT for Admin review before anything leaves the LAN (2026-09-09, final update 19:50: r3 image = fork re-merge + vision; prefill study P1–P6 complete).**
All numbers below were measured today on our 8-node TP8 fleet.

## 1. What was built

| Component | Value |
|---|---|
| vLLM base | upstream **v0.29.0** (tag, 2026-09-08) |
| Fork merged | local-inference-lab/vllm `dev/jovian-judgement` @ `f9dc27dde` (2026-09-07: DeepSeek V4 vision + B12x runtime fixes) |
| Our branch | `botlabs21/v0.29.0rc6-b12x` @ `13473c731` = v0.29.0 + 4 commits (fork merge, drop fork-only Qwen3.8 files, 3 merge-slip fixes, v0.29.0 merge) |
| Our branch, **r3** (later the same day) | `9e7864d3d` = the above + the fork's 3 newer commits cherry-picked (`aba94d396` GLM-5.3 DSpark adaptive verification, `09aa9ecaa` router BF16 reduction ownership, `41ea64ae6` loader-independent mmap offload) + **GLM-5.3 vision port** (`Glm5vForConditionalGeneration`, 3 commits) |
| Delta vs v0.29.0 | 226 files, +32.6K / −1.5K lines (the fork's b12x glue: sparse-MLA + indexer backends, RoCE/PCIe all-reduce, V2 model tree for DSA models, GLM/DeepSeek profiles) |
| b12x | `lukealonso/b12x` @ **`75ffee63`** (1.3.0), built from source, **pinned** — master (`4774ca8e`) already requires fork symbols newer than our merge base |
| b12x, **r3** | `4774ca8e` — matches the re-merged fork head; the `b12x_loader` plugin (InstantTensor/fastsafetensors loading) now imports cleanly |
| torch / CUDA | 2.13.0+cu130 / 13.0 (sm_121a) |
| FlashInfer | 0.6.18 (`866acb62`), + targeted vLLM PR #47392 runtime patch (pipeline default) |
| NCCL | nvidia-nccl-cu13 2.29.7 |
| transformers / triton / CUTLASS DSL | 5.17.0 / 3.7.1 / 4.7.0 |
| Build pipeline | eugr `spark-vllm-docker` `build-and-copy.sh`, local source dir mode, two flags we added: `--with-b12x` (the pipeline skips the b12x stage for local-source builds), `--b12x-ref <commit>` |
| Build time | ~20 min on one GB10 with warm ccache (first full build 45 min) |
| Image | `vllm-node-botlabs21-0.29.0` (24.4 GB) |
| Image, **r3** | `vllm-node-botlabs21-0.29.0-r3` (id `9b754285206c`) — re-merge + vision; the two post-build vision fixes (`35303cff3`, `9e7864d3d`) were run as overlays on this image and are in the branch for the next bake |
| Image, **r4** (go-forward) | `vllm-node-botlabs21-0.29.0-r4` (id `6a1e008c35d6`) — r3 source at `9e7864d3d` with both vision fixes baked in, **no overlays**; on all 8 brain nodes. Every result from the FRESH row onward ran on it. |

Build command (on the build node, inside the `spark-vllm-docker` checkout):

```
./build-and-copy.sh --vllm-source-dir /mnt/glm52/port/vllm-src -t vllm-node-botlabs21-0.29.0 \
  --with-b12x --b12x-ref 75ffee6375b0577ce2c8d6931ffacefda3ecbdd6 --full-log
```

### Merge notes worth sharing
- 37 conflicts / 246 clean when applying the fork delta onto v0.29.0rc6; conflict decisions in `BOTLABS21_MERGE_NOTES.md` in the branch. Upstream had already absorbed several fork features under other names (Qwen3_8FlashNext → Qwen4Exp, retention interval, boundary offloads, HYV4 in the V2 runner lists).
- **`py_compile` is not a merge gate.** The first TP8 boot died on `NameError: layer_spec` in `vllm/v1/worker/gpu/model_runner.py` (upstream's `layer_spec` / `slot_mapping_enabled` definitions were dropped in the KV-group loop while their consumers were kept). `pyflakes` over the 199 merged `.py` files found two more undefined names: a lost `round_up` import in `v1/core/kv_cache_utils.py` (fork line) and a lost `sample_src_positions_ptr` kernel parameter in both Triton kernels of `v1/worker/gpu/spec_decode/autoregressive/speculator.py` (upstream param sat next to the fork's mrope params). All three fixed in commit `5b275ec45`. Run `pyflakes $(git diff --name-only <base> HEAD -- 'vllm/*.py') | grep "undefined name"` before building any merge.
- **Re-merging the fork later:** if the first fork merge was applied as a squash (ours was), `git merge <fork-head>` replays the whole
  delta (51 conflicts for 3 commits). Cherry-pick the new range from the last merged fork commit instead
  (`git cherry-pick <last-merged>..origin/dev/jovian-judgement`); conflicts then land only in fork-only files.
- The current fork has **no `B12X_MLA_SPARSE` enum member**: use `--attention-backend B12X`; `vllm/models/deepseek_v32/nvidia/model.py` routes DSA models (GLM-5.2/5.3, DeepSeek V3.2) to `B12xGLMDSAMLASparseBackend`. `B12X_MLA_SPARSE` in older recipes (incl. the shipped `recipes/8x-spark-cluster/glm-5.2-nvfp4.yaml`) is stale.
- `adaptive_speculative_tokens_window` now **requires** `num_speculative_tokens_per_batch_size` (else the worker raises "num_speculative_tokens_per_batch_size is required for dynamic speculative decoding").
- The fork dropped `--enable-decode-aware-prefill` & co; the stock levers are `prefill_schedule_interval`, `max_num_scheduled_tokens`, `max_num_partial_prefills`.
- V2 speculator note: "Fused multi-step draft decode is not supported by attention backend(s) B12X, B12X_INDEXER; falling back to rebuilding attention metadata between draft steps" — a perf lever for anyone with k>1 on B12X.

## 2. Model & serving shape

- Body: our GLM-5.3 Int4/Int8-mix GPTQ (Pollard method; experts int4 g128, attention/shared int8; 369 GB, text-only repack).
- Drafter: MTP layer 78, int8 (`SAFEL78`), loaded from a separate dir via `speculative-config.model`, `quantization: compressed-tensors`.
- TP8, DCP1, 8× GB10 over a 100G RoCE fabric (one HCA per node), multiproc executor (`run-recipe --no-ray`).
- KV: `nvfp4_ds_mla`, `--max-model-len 900000`, `--gpu-memory-utilization 0.79` → 1.04–1.07M-token pool (1.15–1.19× at 900K); with `--kv-cache-memory-bytes 42GiB` at gmu 0.81 → 1.41M tokens (1.57×). Note: the fork's NVFP4 KV here runs **without** our calibrated outer scales (that loader is not in this tree yet), so quality numbers are not comparable to our production stack.

Production-shape recipe (env + command; run through eugr's `run-recipe.py`):

```yaml
container: vllm-node-botlabs21-0.29.0:latest
model: /root/models/glm53-int4-int8mix-v4-fast-TEXTONLY-0280
env:
  VLLM_USE_V2_MODEL_RUNNER: "1"
  VLLM_USE_B12X_SPARSE_INDEXER: "1"
  VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE: "1"
  VLLM_B12X_MLA_CKV_GATHER: "1"
  VLLM_ENABLE_ROCE_ALLREDUCE: "1"
  VLLM_ROCE_ALLREDUCE_MAX_SIZE: "2MB"
  VLLM_SPARSE_INDEXER_MAX_LOGITS_MB: "256"
  VLLM_ALLOW_LONG_MAX_MODEL_LEN: "1"
  VLLM_SLEEP_WHEN_IDLE: "1"
  CUTE_DSL_ARCH: "sm_121a"
  TORCH_CUDA_ARCH_LIST: "12.1a"
  # NCCL over RoCE: NCCL_NET=IB, NCCL_IB_HCA=<your HCA>, NCCL_IB_GID_INDEX=3, NCCL_IB_TC=106, NCCL_IB_QPS_PER_CONNECTION=2,
  # NCCL_IB_SPLIT_DATA_ON_QPS=1, NCCL_MIN/MAX_NCHANNELS=4, NCCL_BUFFSIZE=16777216, NCCL_NET_GDR_LEVEL=5, NCCL_DMABUF_ENABLE=1,
  # NCCL_CUMEM_ENABLE=0, NCCL_NVLS_ENABLE=0, NCCL_CROSS_NIC=1, NCCL_IB_TIMEOUT=22
command: |
  vllm serve /root/models/glm53-int4-int8mix-v4-fast-TEXTONLY-0280 \
    --trust-remote-code --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
    --enable-chunked-prefill --enable-prefix-caching --async-scheduling \
    --tensor-parallel-size 8 --pipeline-parallel-size 1 --decode-context-parallel-size 1 --dcp-kv-cache-interleave-size 1 \
    --attention-backend B12X \
    --hf-overrides '{"use_index_cache":true,"index_topk_pattern":"FFFSSSFSSS…"}' \
    --max-model-len 900000 --max-num-seqs 5 --max-num-batched-tokens 8232 --long-prefill-token-threshold 2048 \
    --gpu-memory-utilization 0.79 --kv-cache-dtype nvfp4_ds_mla \
    --distributed-executor-backend ray \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,3,4,5,6,8,9,10,12,15,16,18,20,24,25,30,32,40],"max_cudagraph_capture_size":40,"inductor_compile_config":{"combo_kernels":false,"benchmark_combo_kernel":false,"triton.autotune_at_compile_time":true}}' \
    --generation-config vllm --default-chat-template-kwargs '{"reasoning_effort":"high","clear_thinking":false}' \
    --speculative-config '{"model":"/root/models/<mtp-layer-78-dir>","method":"mtp","quantization":"compressed-tensors","attention_backend":"B12X","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,1,5],[2,3,3],[4,64,2]],"draft_sample_method":"probabilistic","adaptive_speculative_tokens_window":32,"draft_tensor_parallel_size":1}' \
    --seed 42
```

## 3. Results (2026-09-09, TP8, every leg: correctness probes ALL PASSED incl. concurrent ×4/×8, 0 errors, boot 9–11 min)

Bench = our mixed-workload generator (agentic/tool 30 %, code 25 %, prose 20 %, long-context 15 %, short-QA 10 %; identical seeded
workload every run; 2-min warm-up, then 300 s at concurrency 1 and 300 s at concurrency 4; aggregate tok/s).

| Leg | KV | RoCE all-reduce | Spec decode | C1 tok/s | C4 tok/s | Accepted/step (rate) |
|---|---|---|---|---|---|---|
| Engine baseline | fp8_ds_mla 600K | off | off | 24.1 | 58.4 | — |
| NVFP4 KV | nvfp4_ds_mla 900K | off | off | 23.9 | 55.9 | — |
| + RoCE | fp8 600K | **on** | off | 32.4 | 67.5 | — |
| + MTP k3 | fp8 600K | on | MTP k3 | 49.8 | 85.3 | 1.82 (60.6 %) |
| Production shape | nvfp4 900K | on | MTP k3 | 49.4 | 82.3 | 1.84 (61.4 %) |
| + k5 schedule + adaptive | nvfp4 900K | on | k5 sched | **51.0** | **85.1** | 1.73 (63.6 %) |
| + humming linear backend | nvfp4 900K | on | k5 sched | 51.2 | 84.4 | 1.72 (63.1 %) |
| gmu 0.81 + 42 GiB KV pin | nvfp4 900K (1.41M pool) | on | k5 sched | 50.0 | 84.4 | 1.61 (61.5 %) |

Reference points: our previous production stack (July fork image + our patches, MTP k5 + n-gram chains, NVFP4 KV 900K) = 40.0 / 85.2 on the same bench; our stock-0.28 port line best = 22.1 / 47.1.

Take-aways: RoCE one-shot all-reduce = +34 % C1 / +16 % C4 on its own; MTP is the other half; KV dtype and pool size do not move
throughput; humming is neutral on this tree (upstream Marlin caught up). The drafter works natively (no head-keeping patch needed).

**Prose probe (community protocol: 10 prose prompts, 250 out, wall incl. TTFT; on the k5 + humming leg):**

| Metric | This stack | Community 8× Spark reference (Sep 8) |
|---|---|---|
| Single-stream prose, median | **42.8 tok/s** (40.1–46.7) | 38.8 |
| Concurrency 2 / 4 / 8 aggregate | 62.0 / 86.3 / **121.7** | — / — / 110.7 |
| Prefill 42K / 92K / 132K prompt | 924 / 877 / 844 tok/s | 1.27–1.41K |

Prefill is the one metric we trail (~1/3), and it decays with length; legs P1/P2 below target it.

Long-context: needle probe at 634K prompt tokens (two planted passphrases at 30 %/80 % depth) → both found, 1135 s wall (~560 tok/s
prefill at that length), engine and all 8 nodes healthy afterwards, at gmu 0.81 + 42 GiB pin.

## 4. Prefill legs (on the pinned image `63cf60f5acbb`, no overlays — these two boots also confirm the baked fixes)

| Leg | C1 | C4 | Acc/step | Prose single | c8 | Prefill 42K / 92K / 132K tok/s |
|---|---|---|---|---|---|---|
| Reference (leg 7, CKV gather on, batched 8232) | 51.2 | 84.4 | 1.72 | 42.8 | 121.7 | 924 / 877 / 844 |
| P1: `VLLM_B12X_MLA_CKV_GATHER=0` | 50.4 | 85.0 | 1.70 | 42.4 | 119.5 | 903 / 859 / 828 |
| P2: `--max-num-batched-tokens 16384` | 50.3 | 83.7 | 1.71 | 43.3 | 121.2 | 904 / 859 / 828 |
| r3 text (fork re-merge, after drop-caches + a manual full compaction on the nodes) | 50.8 | 85.3 | 1.60 | 42.3 | 120.8 | 906 / 861 / 830 |
| P3: `--moe-backend triton` (Triton `moe_wna16` instead of Marlin) | 18.8 | 21.9 | 1.79 | 16.7 | 34.7 | 545 / 527 / 514 |
| P4: NCCL_MIN/MAX_NCHANNELS 8/16 + indexer logits 1024 MB + VLLM_USE_AOT_COMPILE + MEGA_AOT | 50.5 | 84.0 | 1.77 | 42.2 | 116.7 | 862 / 822 / 796 |
| **P6: `--kv-cache-dtype fp8_ds_mla` @ 600K** (everything else the production shape) | 39.0* | 86.1 | 1.74 | **43.8** | **128.8** | **1007 / 942 / 902** |
| FRESH: same recipe as r3 text, 10 min after a full fleet reboot (uptime 21 d → 0) | 52.3 | 85.8 | — | 43.8 | 123.2 | 909 / 865 / 833 |
| CHAMPION (production stack: ciprianveg v18.1 image, GPTQ int4/int8 v4-fast, MTP SAFEL78, NVFP4 KV @900K) measured 15 min after the FRESH leg on the same rebooted fleet | — | — | — | 33.9 | 126.3 | **960 / 930 / 903** |

The CHAMPION row is the control the prefill legs were missing: on the same fresh fleet the older production stack prefills 5–8 % *faster*
than the 0.29.0+b12x port (960 vs 909 at 42K, 903 vs 833 at 132K) while decoding 25 % slower on prose (33.9 vs 43.8). So the port's
prefill deficit against the champion is a property of the new stack (b12x v19 / vLLM 0.29 prefill path), not of the fleet, and it is
smaller than the decode gain. Tracked as an open gap; the fp8-KV leg (P6) already recovers it and then some.

\* single mixed-bench C1 sample on 35 requests; the prose single-stream (10 runs) on the same engine was the best of the day.

The r3 row also rules out two things at once: the re-merged fork commits are neutral, and clearing page cache plus a manual
full compaction on the nodes (the fragmentation theory from the community thread) does nothing here — with the engine torn down our
nodes already hold ~5,000 order-10 free blocks, so the "no high-order blocks" reading only ever described the loaded state.

P3 settles the MoE-kernel question the other way: on GB10 the Triton `moe_wna16` kernel is ~40 % slower than Marlin on prefill
and ~3x slower on decode (acceptance unchanged, so it is pure kernel speed) — upstream's "Triton wins at large M" result does not
hold on sm_121. Marlin stays for both phases.

P4 (more NCCL channels for the ~100 MB prefill all-reduces, a 4x larger sparse-indexer logits budget, AOT compile) is ~5 % *worse*
on prefill with decode unchanged — on a single 100G port, 4 channels was already right.

**P6 is the one lever that moved prefill: fp8 KV is +11 % on prefill over NVFP4 KV at every length**, and gave the best prose
single-stream and c8 numbers of the day. The NVFP4 write path quantizes every token's 512-dim latent with a per-token scale during
prefill; that is the cost. The trade is context: fp8 at gmu 0.79 holds a 656K-token pool (600K context), NVFP4 holds 1.04–1.07M
(900K). After P6 the remaining ~28 % to the reference stack is not in any software knob we could reach: the same fork lineage on a
**200G dual-port** fabric (our second RoCE port is uncabled), a fresh-boot fleet (their prefill halves within a day of uptime; ours
has 21 days and no before/after measurement), and RTN vs GPTQ-g128 Marlin shapes are what is left.

**Uptime/fragmentation: ruled out on this fleet.** The community thread reports prefill halving within a day of uptime and
recovering only on reboot. We rebooted all eight nodes (21 days up → fresh) and re-ran the identical recipe: 909/865/833 vs
906/861/830 before — no effect at all. Idle nodes here also show thousands of order-10 free blocks; the "no high-order blocks" state
is just the loaded engine. Their decay must come from something in their environment (container --memory cap, host memory pressure)
that we do not share.

Neither knob moves prefill: CKV gather off is −2 % (noise), and doubling the chunk from 8232 to 16384 tokens changes nothing at all
(904/859/828 vs 903/859/828). Prefill on this stack is bound by the sparse-MLA / indexer prefill kernels, not by scheduling — the
~1/3 gap to the community's 1.27–1.41K figure has to come from the kernel path (or from their measurement using shorter prompts),
not from scheduler flags. Decode and acceptance are flat across all three, so the production shape keeps CKV gather on and 8232.

## 4b. GLM-5.3 vision on the V2 tree (r3) — CONFIRMED 2026-09-09 17:11

The July fork overlay's `Glm5vForConditionalGeneration` (MoonViT tower + Kimi-K2.5 PatchMerger projector, from the ciprianveg v18.1
image) ported onto this tree over the V2 `GlmMoeDsaForCausalLM`. Upstream already carries the Kimi-K2.5 vision code (`kimi_k25.py`,
`kimi_k25_vit.py`, processor, `KimiK25VisionConfig`), so the port is a 200-line wrapper + a 100-line config + four registrations.
Two runtime lessons for anyone wrapping a V2 text model:
- the V2 speculator builds the MTP draft from the **target's** `hf_config`; a vision wrapper config must delegate unknown attributes
  (`n_group`, …) to its `text_config`, or the draft load dies with `AttributeError`;
- the V2 runner treats the *presence* of `get_mtp_target_hidden_states` as a promise of a tensor (it slices the result) — only expose
  the hook when the text model has it (DeepSeek-V4 does, GlmMoeDsa does not).

Result on the composite checkpoint (text + `vision_tower.*`/`mm_projector.*`, `--limit-mm-per-prompt image=4`, encoder tp-mode data,
same spec/KV/RoCE as the production shape): boot 11 min, correctness ALL PASSED, **image probe 2/2** (red square with a "7", green
circle — both described correctly, ~130 prompt tokens each), text C1 50.2 / C4 84.2, acceptance 1.76/step (62.5 %) — i.e. the vision
tower costs nothing on text throughput and the MTP drafter works with the VL target.

## 5. Credits — this is assembled on other people's work

- **Z.ai (zai-org)** — GLM-5.3 itself: the 744B/40B-active MoE body and its MTP layer, which is the drafter in every spec-decode leg
  here. Everything below is machinery for serving their model.
- **DeepSeek** — the MLA + DSA sparse-attention architecture (index_topk, kv_lora_rank 512 + rope 64) that GLM-5.3 uses and that the
  b12x sparse-MLA/indexer kernels, the `fp8_ds_mla` / `nvfp4_ds_mla` KV layouts and the fork's `deepseek_v32` model tree implement.
- **GPTQ** (Frantar, Ashkboos, Hoefler, Alistarh — IST-DASLab) — the Hessian-based quantization algorithm under the Pollard method
  that produced the Int4/Int8 body.
- **vLLM** (vllm-project, Apache-2.0) — the engine; v0.29.0 is the base of this branch.
- **local-inference-lab/vllm `dev/jovian-judgement`** — the DGX Spark fork whose delta we merged onto v0.29.0: the B12X sparse-MLA and
  DSA-indexer attention backends, the RoCE/PCIe one-shot all-reduce collectives, the V2 model tree for GLM/DeepSeek DSA models, the
  GLM-5.2/5.3 profiles and the MTP speculator path. Top committers of that delta: Luke Alonso, Martin Vit, derek, logprobz, MadeBy561.
  Every throughput number in §3 stands on this code; our contribution is the merge, three merge-slip fixes and the measurements.
- **b12x** (Luke Alonso, Apache-2.0) — the GB10 kernel package (sparse MLA, indexer, MoE/GEMM lanes, RoCE collectives) that the fork
  drives; built from source at `75ffee63`.
- **eugr/spark-vllm-docker** (MIT) — the build pipeline (NCCL + FlashInfer + vLLM + b12x from source, patch scripts, `run-recipe`
  cluster launcher and recipe format) and the reference multi-node recipes. We added two flags (`--with-b12x`, `--b12x-ref`) and
  made the b12x clone accept a commit SHA; the pipeline is otherwise theirs.
- **yichengj0** — vLLM PR #47392 "[Bugfix][MoE] Plumb swigluoai activation into FlashInfer b12x MoE" (open upstream), whose runtime
  subset the pipeline applies as a patch to every build.
- **ciprianveg/gb10-glm-5.2 and gb10-vllm** — the v18.1-vision image (July fork + b12x 0.30.2 glue, humming profile, Glm5v vision,
  DFlash2 chain drafting) and its recipes are what we have run GLM-5.2/5.3 on in production for the past two months, and the
  starting point every recipe in this report descends from; the "previous production stack" reference numbers in §3 are that image
  plus our patches. Our GB10 port findings were shared back as issues on gb10-vllm (#3 port contracts, #4 GB10 ops).
- **FlashInfer**, **NVIDIA CUTLASS DSL**, **Marlin** (IST-DASLab; the WNA16 kernels the int4/int8 body runs on), **NCCL**.
- **humming** (inclusionAI, `humming-kernels` 0.1.12) — the quantized-GEMM kernel package behind `--linear-backend humming`
  (neutral in our measurement here, +16–24 % C4 on the July fork); it in turn builds on Marlin, DeepGEMM (deepseek-ai), lmdeploy
  (InternLM) and CUTLASS.
- **Pollard Weights** (WestWaters) — the quantization method behind the Int4/Int8-mix body (Hessian GPTQ + sensitivity-driven
  int4/int8 allocation); our routing-concentration and norm-seam data went back to that project as PR #36.
- **QuantTrio** — the original GLM-5.2 Int4-Int8Mix checkpoints; the int4-experts / int8-attention layout our body follows (and the
  served-model alias we still carry) comes from their work. **Tech2wild / tonyd2wild** — the GLM-5.3 Int4-Int8Mix TP4 recipes and the
  GLM-5.2 QuantTrio 200K 4× Spark recipe that were our reference for the GB10 serving flags before we had our own numbers.
- **Light Foundry Notes (@light_foundry on X)** — the Sep 8 8× Spark GLM-5.3 Int4/Int8Mix post whose 38.8 tok/s prose /
  110.7 c8 / 1.27–1.41K prefill figures are the reference column in §3, and whose knob list (RoCE all-reduce, MTP over DFlash2,
  CKV_GATHER=0) shaped our leg order.
- The DFlash / DFlash2 authors and the MTP drafter fine-tuning work (ours) are not part of this image yet; they are credited where
  those pieces land.

## 6. Known gaps / not in this image yet
- The n-gram lookup + chain drafting we run in production (~2.5x on verbatim-heavy turns) — still to port onto the V2 speculator.
- Not needed after all (verified 09-09): our calibrated NVFP4 outer scales (this tree's b12x uses a per-token fp32 latent scale, the
  per-layer scalar is folded out), the fp8-rope writer (subsumed by the fork's NVFP4 layout), the decode-aware prefill scheduler (C4 matches),
  humming (neutral). Vision: ported (§4b).
- Prefill: ~30 % below the @light_foundry figures at comparable lengths; scheduler knobs are inert (§4). Levers still open: Triton
  `moe_wna16` MoE backend instead of Marlin for large-M prefill GEMMs, NCCL channel count for the 100 MB prefill all-reduces (both
  sides pin 4), sparse-indexer logits budget, AOT compile, MLA prefill backend, and cabling the second RoCE port (both fleets run one).
- b12x master moves with the fork daily; pin it (`--b12x-ref`) or re-merge the fork head first.
