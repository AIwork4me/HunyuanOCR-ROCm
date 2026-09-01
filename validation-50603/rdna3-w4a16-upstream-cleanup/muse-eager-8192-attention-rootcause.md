# Muse eager ctx8192 残留 2/8 的根因诊断报告（2026-09-01）

本报告记录 vllm-50603-upstream-cleanup 工作区对 HANDOFF 第四节唯一未关闭问题的完整诊断过程与结论。所有实验均在 W7900D（gfx1100）、env-clean（vllm-clean @ 07ea9350 + upstream-final.patch）上真实运行。

## 结论（TL;DR）

残留的 Muse eager ctx8192 = 2/8（gen0 与 gens1-7 两族，first_div 恒为 27）**不是 W4A16 patch 的确定性问题**：逐调用拦截证明补丁后的 GEMM kernel 在全部 16640 次调用中 0 次出现"相同输入不同输出"。分叉首次出现在 vLLM 上游的 **ROCM custom paged attention kernel**（NoPE/full-attention 层的 decode 路径）的输出张量：1/4096 个元素相差约 1.5 ULP。该 kernel 的输出被证实依赖 KV cache 末块中 seq_len 之后的 stale 槽位内容（实验 I 的整块交换直接改变输出），而 stale 内容由分配器历史决定（512 相位先行、M=16 prefill 走 v1-WMMA 还是 scalar 的 partials 尺寸差异等）。这与"graphs 模式 1/8、直接 8192 不复现、M>=65 dispatch 下 1/8"全部现象吻合。该缺陷位于 vLLM 上游 attention kernel，应作为独立 issue 上报；对本 PR 的 E2E 声明只需如实记录。

## 现象与复现

两段探针（`probe_two_phase.py`）：同引擎先跑 ctx512 段（warmup + 8×64 token），再跑 ctx8192 段。结果 512=1/8、8192=2/8，与 HANDOFF 及 e2e 两个引擎的结果一致（first_div 全部=27）。只跑 8192 段不复现（1/8）；graphs 模式 1/8；gemma 全 1/8。

关键结构事实（拦截记录实测）：由于 prefix caching，8192 段每个 gen 的 quant 调用 = 每层 1 次 M=16（末块重算 prefill）+ 63 步 M=1 decode；不存在 M=8192 的 quant 调用（warmup 除外）。Muse 为 iRoPE 架构：RoPE 层=sliding window（2047），NoPE 层=full attention；NoPE 层 decode 走 `ops.paged_attention`（csrc/rocm/attention.cu 的 ll4mi kernel，通过 `use_rocm_custom_paged_attention` 选择，要求 sliding_window 无效值），sliding 层走 Triton decode kernel。

## 证据链（按探针顺序）

1. **quant 拦截（probe_intercept）**：记录全部 16640 次 `gptq_gemm_rdna3` 的输入/输出哈希，g0 vs g1 逐调用对比：**kernel 级分歧（输入同、输出异）= 0 次**。首个分叉为 o_proj 输入（第 1837 调用，step7/layer3，M=1 N=6656 K=4096）——分叉从 attention 区进入，非 quant kernel。
2. **attention 拦截（probe_attn）**：记录每次 `chunked_prefill_paged_decode` 的 q/o 哈希：g1 vs g2 全等；g0 vs g1 的输出差异全部伴随 q 已差异（该轮分叉在更上游的 glue 区域，位置随插桩漂移——本身即"运行状态依赖"的信号）。
3. **全链路拦截（probe_chain2，修复 gemm shim 遗漏后）**：合并记录 gemm/全部 RMSNorm/rope/attention/DecoderLayer/Attention/MLP 块共 50048 条：首个分叉 = `sigmoid(gate) * attn_output` 的乘积（gate y 与 attention o 在各自哈希时刻完全一致）。
4. **细粒度复刻（probe_fine）**：复刻 MuseGlimmerAttention.forward 逐步哈希：首个分叉 = layer3（NoPE）attention 输出生产时刻；加每步全设备栅栏（FULLSYNC 变体）结果不变——排除跨流异步写。
5. **KV 内容哈希（probe_kvv）**：q、有效 KV（全部完整块）在分叉点完全一致；attention 输出首个分叉 = 1/4096 元素、|Δ|=1.2e-4（~1.5 ULP @0.02）、无 NaN；末块（含 stale padding）哈希从第一条记录即不同。
6. **同输入重复（probe_rep）**：1638 次"同参数立即重调"零差异——排除简单竞态。
7. **末块整块交换（probe_stale）**：把 g0 的末块写入 g1 的 cache 后用相同 q/table/seq 重调，819 次交换后输出无一匹配 g0 的原输出——即使两 gen 输出本相同的调用，交换末块也改变输出。直接证明输出依赖末块内容（有效槽相同 ⇒ 实质是 stale 槽）。
8. **dispatch A/B**：临时把 dispatch 改回 M>=65（M=16 走 scalar，旧 build 数值路径）重建：两段探针 512=1、8192=1；attention 拦截全等。恢复 M>=16 后 2/8 复现。M=16 的 v1-WMMA 路径输出本身逐位确定（证据 1），其因果角色是 per-call partials 分配改变了分配器布局时间线，从而改变 attention 末块 stale 内容。
9. **attention 工作区 zeros（阴性对照）**：将 custom kernel 的 tmp_output/exp_sums/max_logits 改为 torch.zeros 不改变结果（仍 2/8）——排除该工作区未写即读。
10. **静态审计**：ll4mi kernel 的 K/V 越界 token 有块号钳制（last_seq_block）且 QK logits 有 `-FLT_MAX` 屏蔽、exp 权重置 0；reducer 只读按 seq_len 界定的 partition。泄漏点未在上游 kernel 内定位到具体行（属于上游调查），但"输出依赖 stale 槽"已被实验 7 直接证明。

## 对各现象的解释

- 只在 512 段先行后复现：512 段改变分配器与 KV 块布局历史，决定 8192 段末块 stale 字节。
- gen0 与 gens1-7 两族且 first_div 恒 27：stale 内容在 gen0（新页/新块）与后续 gen（复用路径稳定）间形成两个确定态；1.5 ULP 级差异通常不跨过 argmax，恰好在 token 27 处跨越（两引擎一致）。
- graphs 1/8：capture pool 的分配布局不同，stale 态一致。
- M>=65 旧 dispatch 1/8：M=16 走 scalar partials（尺寸/填充不同），stale 态不同，未跨越 argmax。
- 直接 8192（无 512 段）1/8：分配历史不同。

## 处置建议

1. 本 PR（W4A16 determinism）保持 upstream dispatch 与当前 epilogue 不变；E2E 记录：Muse graphs 512/8192 均 1/8、eager 512 1/8、gemma eager 1/8；Muse eager 8192 = 2/8 已定位为上游 ROCm custom paged attention 的 KV 末块 stale 槽敏感性问题（本报告），与本 patch 无关（kernel 级 0 分歧证据）。
2. 向 vLLM 上游另开 issue：附最小化证据（probe_stale 的整块交换复现方法、probe_kvv 的 1-ULP 输出 diff、NoPE/GQA/长上下文触发条件：sliding_window 无效值层 + seq_len 跨 256-token partition 边界 + 末块未填满）。
3. 如需在本地规避：attention_backend=TRITON_ATTN 的两段探针结果：**512=1、8192=1**（logs/probe-triton.log）——同一 W4A16 patch、仅换 attention backend 即完全确定，是"残留与 patch 无关"的最直接端到端证据，也是上游 issue 的规避路径。

## 工件清单

- 探针脚本：probe_two_phase.py / probe_intercept.py / probe_attn.py / probe_chain2.py / probe_fine.py / probe_kvv.py / probe_stale.py / probe_rep.py / probe_backend.py / probe_micro.py（微测隔离，未再需要）
- 日志与 JSON：logs/probe-*.log、logs/intercept-g*.json、logs/attn-g*.json、logs/chain2-g*.json、logs/fine-g*.json、logs/kvv-g*.json、logs/stale-g*.json、logs/rep-g*.json
- 临时改动均已还原：dispatch 恢复 M>=16（重建验证）、chunked_prefill_paged_decode.py 的 zeros 实验已撤销（git diff 干净）。
