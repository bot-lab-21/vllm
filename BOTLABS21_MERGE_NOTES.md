# bot-labs-21 merge: local-inference-lab/vllm dev/jovian-judgement (f9dc27dde) onto v0.29.0rc6 — 2026-09-08

Base: upstream v0.29.0rc6 (74c96922e). Fork delta = `git diff 299ebd094..dev/jovian-judgement` (merge-base with upstream main,
2026-08-25), applied with `git apply -3 --exclude='tests/*'`: 246 files clean, 37 conflicted, resolved as below. Tests excluded
on purpose (re-add per area when the build runs).

Rules used: where upstream had since merged a fork feature under another name (Qwen3_8FlashNext → Qwen4Exp, retention
interval, boundary offloads), keep upstream's names; where the fork's cleanly-applied code depends on fork names
(hit_alignment_tokens, boundary checkpoint cache, packed nvfp4 KV update, connector block-state builder, group CP sizes),
take the fork; where both are independent features, keep both.

- rc6 side (dropped fork variant): Qwen3_8FlashNext naming in config/registry/transformers_utils/models/config.py;
  kv_cache_utils (DeepseekV4 grouping + GLM-5.3-Flash split cache pages: `VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE` path dropped);
  flash_attn.py (fa4 hd256 + rc6 DCP init; fork DFlash fast path dropped); cudagraph_utils hunk 3 (rc6 dense SD schedule);
  gpu_model_runner.py mamba copy funcs; mamba_utils.py, model_states/mamba_hybrid.py, qwen_gdn_linear_attn.py, mooncake store,
  kv_events.py (out of scope for GLM-5.3); spec_decode/speculator.py HC-residual widening (rc6 hook-based).
- fork side: kv_cache_coordinator + single_type_kv_cache_manager (hit_alignment_tokens, retention_eagle_rewind);
  kv_cache_manager (BoundaryCheckpointCache); kv_cache_interface (page_tail_bytes_per_token, dcp_replicated);
  sparse_utils NUM_TOPK_TOKENS; deepseek_v32 attention/kernels (`_native_packed_kv_update`, mla_cache_block_size);
  scheduler `_build_kv_connector_block_state` + sched/output.py; gpu_worker.py late-persistent-memory profiling;
  cudagraph_utils hunks 4-6 (capture resources, fork profiling with `_teardown_profiling_state`, unbind_kv_cache);
  block_table.py + model_runner.py group_cp_sizes; model_runner `profile_glm_dcp_attention`;
  autoregressive speculator hunks 2-6 (prefill_outputs_are_compact, mrope positions).
- both (merged by hand): arg_utils imports (FairnessEngine + get_from_deprecated_env_if_set) and cache kwargs
  (prefix_cache_retention_interval=retention_interval + recurrent_checkpoint_policy); config/vllm.py MTP arch list
  (HYV4 + Glm5Next); scheduler `has_sync_kv_loads` + `scheduled_prefill_req_ids`; worker/utils.py `clear_layer_kv_caches`
  + `unbind_kv_cache`; cudagraph_utils imports; speculator signatures carry BOTH `dp_sync` (rc6) and
  `num_speculative_tokens`/`num_tokens_across_dp` (fork) — call site in model_runner passes both; block_table.py carries
  BOTH `group_cp_sizes` (fork) and `slot_mapping_enabled` (rc6) through constructor, tensors and the Triton kernel;
  vocab_parallel_embedding accepts BOTH `quant_method` (rc6) and `lm_head_quantization` (fork) and uses rc6's fused-embedding
  flag with the fork's `allocate_weights(...)` allocation.

Known follow-ups before/while building: dflash speculator `draft_cp_size` hunk dropped (fork DCP draft; re-add if DFlash on
DCP>1 is wanted); fork `vllm/models/qwen3_8_flash_next/` files are present but unregistered (upstream has qwen4_exp) — delete
before build; boundary_checkpoint.py uses block_tables.group_cp_sizes (present). Our own bake list (fp8-rope KV writer,
nvfp4 outer-scale loader, C2 fix, mtpfix, keep-trained-head gate, spin-wait, humming) is NOT yet applied.
