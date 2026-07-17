# Visual design brief — README hero & Social Preview

Status: **design-only**. This brief specifies the README hero image and the
GitHub Social Preview so a maintainer (or designer) can produce final assets. No
placeholder image is committed — cheap icon collages or unlicensed brand marks
are explicitly forbidden (see "Forbidden").

## Why no image yet

This environment cannot produce a high-quality, on-brand hero. Committing a
low-effort placeholder would lower perceived quality. The image is a tracked
maintainer to-do, not a blocker for any code gate.

## 1. README hero image

- **Size:** 1280×360 px (≈3.5:1 banner), ≤ 256 KB, PNG or optimized WebP.
- **Core message (one glance):** "HunyuanOCR-1.5 on AMD ROCm — three backends,
  OmniDocBench v1.6." Lead with the headline number **llama.cpp full 1651 =
  92.09** and the status **evaluation-backed, not precision-aligned**.
- **Visual hierarchy:** (1) project wordmark, (2) the headline score + page count,
  (3) three backend chips (llama.cpp / vLLM / transformers), (4) a small
  "gfx1100 · RDNA3 · ROCm" tag. Nothing else competes.
- **Color:** dark slate background (`#0E1116`-ish) with AMD-red (`#ED1C24`) and a
  single accent for the headline number. High contrast for legibility at small
  sizes.
- **Place in README:** directly under the H1 + tagline, above the badges.

## 2. GitHub Social Preview

- **Size:** 1280×640 px (the GitHub OG-image canvas), PNG, ≤ 1 MB.
- **Core message:** project name, "evaluation-backed AMD ROCm port of
  HunyuanOCR-1.5", the 92.09 headline, three backend chips, gfx1100/ROCm tag.
- **Hierarchy:** same as the hero but with more vertical room — larger wordmark,
  the headline score as the focal point.
- **Repo URL footer:** `github.com/AIwork4me/HunyuanOCR-ROCm` in small caps.

## Forbidden

- Do **not** use the Tencent / Hunyuan logo or wordmark (not affiliated; see
  NOTICE). "Powered by Tencent Hunyuan" may appear as small text only if desired.
- Do **not** paste the AMD corporate logo without permission; use a generic "RDNA3
  / ROCm" text tag instead.
- Do **not** use random stock icons, emoji collages, or AI-generated clutter.
- Do **not** claim or imply "precision-aligned" anywhere in the artwork.

## Deliverable checklist (maintainer)

- [ ] Produce hero (1280×360) and Social Preview (1280×640) per above.
- [ ] Store under `docs/visuals/` (or `.github/` for the Social Preview) and link
      from README.
- [ ] Upload the Social Preview in GitHub repo Settings → Social Preview.
- [ ] Keep text claims identical to README (92.09 full; evaluation-backed).
