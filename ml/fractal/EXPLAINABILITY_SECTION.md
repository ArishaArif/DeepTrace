# Explainability — The Fractal Branch

This section explains *how* the fractal branch decides, *why* it looks the way it
does, and *what we verified* about its explanations. It is written to be read
standalone in the report and defended in the viva.

---

## 1. What the branch actually measures

The fractal branch never sees a neural embedding. It measures **texture
self-similarity** directly on pixels, patch by patch, and turns that into a
per-patch "fakeness" probability. The pipeline is:

```
image → grayscale → 64×64 patches (stride 32) → per-patch features → patch
classifier → patch-score matrix → fractal_score (mean) + heatmap
```

Each patch is described by a small, human-interpretable feature vector. The v2
model uses **22 features** (the frozen candidate F). They fall into four
interpretable families:

| Family | Features | Plain-language meaning |
|---|---|---|
| **Box-counting dimension** | `D`, `r2` | How strongly the patch's binarized texture fills space across scales. `r2` is how well the log-log straight line fits — a clean power law means true multiscale structure. |
| **Lacunarity** | `lac2, lac4, lac8, lac16, lac32` | Gappiness / heterogeneity of the texture at five window sizes. Real natural texture has a characteristic decay of lacunarity with scale; GAN/ diffusion artifacts flatten it. |
| **Local slopes** | `slope_small, slope_large, slope_diff` | The box-counting slope measured separately on fine and coarse scales. Real images keep a roughly constant slope; upscaled/generated textures bend it. |
| **Multifractal spectrum** | `D0/D1/D2 grad & int`, `width_grad`, `width_int` | Generalized dimensions of the gradient and intensity measures — how the *distribution* of mass (not just its support) scales. The `D0−D2` width is the classic multifractality spread. |

The mean binarization block (`D_mean, lac4_mean, lac8_mean, lac16_mean`) repeats
the core features under a second thresholding rule so the classifier is not
slaved to one binarization choice.

**Why these and not a deep net:** the whole point of this branch is that every
number fed to the classifier has a physical meaning we can state out loud. That
is what makes the branch explainable *by construction* — the explanation is the
feature vector, not a post-hoc reconstruction.

---

## 2. From patch to heatmap

`predict_fractal(img)` returns `patch_scores`, a `(rows, cols)` float32 array in
`[0, 1]` — one fake-probability per patch — and `fractal_score`, its mean. The
Backend upsamples `patch_scores` into the overlay heatmap; nothing about that
contract changed between v1 and v2.

For a standard 256×256 input this is a 7×7 grid (stride 32). A 128×128 gives 3×3;
anything under 256 on a side is upscaled first, so the grid is never degenerate.

---

## 3. Verification of the explanation

A heatmap is only an explanation if we show it tracks something real. We ran two
families of checks on the **frozen v2 model** over the full validation split.

### 3.1 Quantitative heatmap diagnostics (4,000 val images)

| Check | Real | Fake | Reading |
|---|---|---|---|
| Heat concentration in top-20% patches | 0.232 | 0.227 | Both only slightly above the 0.20 uniform baseline — evidence is **distributed**, not concentrated in a few hot tiles. |
| Center vs border mean score | 0.497 / 0.478 | 0.515 / 0.508 | Mild center bias for both classes; the branch is not fixating on frame edges. |
| Flat heatmaps (std < 0.02) | — | — | **0.15%** of images — almost every image carries genuine spatial variation. |
| Correlation with mean brightness | — | — | **−0.068** |
| Correlation with contrast | — | — | **+0.017** |

The last two are the important confound guards: near-zero correlation with global
brightness and contrast means the branch is **not** just reading exposure or
dynamic range. It is responding to scale/texture structure, which is the intended
signal.

### 3.2 Deletion faithfulness test

The standard way to prove a heatmap is causal, not decorative: blur the *hottest*
`k` patches and measure how much the `fractal_score` drops, then compare against
blurring `k` *random* patches. A faithful map should make the hottest deletion
hurt strictly more.

**Result (v2, k = 2, 4, 8, 200 images each, 10 random draws): NOT faithful.**

| k | hottest drop | random drop | difference | 95% CI | verdict |
|---|---|---|---|---|---|
| 2 | 0.2136 | 0.2136 | −0.000004 | [−0.00008, +0.00006] | not faithful |
| 4 | 0.2136 | 0.2136 | +0.000032 | [−0.00003, +0.00010] | not faithful |
| 8 | 0.2136 | 0.2136 | +0.000006 | [−0.00008, +0.00011] | not faithful |

Deleting the hottest patches is indistinguishable from deleting random patches.
This is consistent with the concentration number above: because the score is a
**mean over a whole image's patch grid** and the evidence is spread thinly across
tiles, no small set of "hot" patches owns the decision. Blurring any handful of
patches perturbs the aggregate by roughly the same amount.

**How to report this honestly.** The heatmap is a *diagnostic map of where the
texture statistic is high*, not a proven causal localization. The branch's
explanation lives at the **feature** level (interpretable by construction), not
at the pixel-localization level. We do **not** claim the heatmap pinpoints the
manipulated region; we claim it shows which scales/regions the fractal statistic
responds to, and we verified that response is not a brightness/contrast artifact.
Localization faithfulness is a known limitation and a concrete next step (a
per-patch, non-mean-aggregated deletion, or a contrast-based relevance measure,
would be the fix).

---

## 4. Why v2 is more trustworthy than v1, with numbers

Same classifier family, same aggregation, same frozen interface — only the
feature vector grew from 4 to 22 interpretable features.

| Metric (val, 4,000 images) | v1 | v2 |
|---|---|---|
| Image AUC | 0.6934 | **0.7802** |
| Patch AUC | 0.5573 | 0.6020 |
| Image accuracy @ 0.5 | 0.6405 | **0.7050** |
| Train CV image AUC (GroupKFold 5) | 0.6867 | **0.7691** |
| Warm inference (256×256, CPU) | 62 ms | 143 ms |

The +0.0824 CV gain is ~9× the fold standard deviation (0.0090), so the adoption
gate's first condition (beat baseline by ≥ 0.01 **and** by > 1 fold-std) passes
comfortably. Inference stays well under the 1 s target, satisfying the gate's
second condition.

Note the still-modest **patch AUC (0.60)** against a much higher **image AUC
(0.78)**: individual patches are weakly informative, but averaging 49 of them
gives a reliable image decision. This is exactly the same effect the deletion
test surfaced — the signal is collective, not local.

---

## 5. Robustness (Step 5)

We re-scored 1,000 val images (500/class, seed 42) through `predict_fractal`
under seven conditions. Image AUC at threshold 0.5:

| Corruption | Image AUC | Accuracy @ 0.5 | Δ AUC vs clean |
|---|---|---|---|
| clean | **0.798** | 0.700 | — |
| JPEG q90 | 0.773 | 0.690 | −0.025 |
| JPEG q75 | 0.638 | 0.515 | −0.160 |
| JPEG q50 | 0.580 | 0.505 | −0.218 |
| resize half→up | 0.586 | 0.500 | −0.211 |
| Gaussian blur σ1 | 0.647 | 0.500 | −0.150 |
| Gaussian noise σ5 | 0.609 | 0.555 | −0.188 |

**Honest reading.** The branch is essentially **immune to mild recompression**
(JPEG q90 costs only 0.025 AUC) but **degrades sharply under anything that
removes high-frequency detail** — aggressive JPEG, half-resolution rescaling, and
blur all fall toward chance-level *accuracy* (~0.50) even while retaining some
*AUC*. This is exactly what the mechanism predicts: box-counting and lacunarity
read fine-scale texture, and resampling/low-pass filtering erase that scale, so
the statistics lose their signal. Noise σ5 hurts less than blur because noise
*adds* high-frequency content rather than removing it.

**Implication for deployment:** input images must reach the branch at native
resolution and high JPEG quality. A pipeline that downscales faces before this
branch would silently disable it — the Deep/Backend owners should feed the
fractal branch a full-resolution crop, not a thumbnail. This is a genuine
limitation of a hand-crafted frequency detector, stated plainly rather than
hidden. Figure: `figures/robustness.png`.

---

## 6. One-paragraph viva summary

> The fractal branch is explainable by construction: it feeds a classifier a
> short vector of physically meaningful texture statistics — box-counting
> dimension and its fit quality, lacunarity at five scales, local multiscale
> slopes, and a multifractal spectrum — instead of an opaque embedding. We verified
> the resulting heatmap is **not** a brightness or contrast artifact (correlations
> −0.07 and +0.02) and carries real spatial variation (only 0.15% flat maps). We
> also report honestly that a deletion test finds the map **not causally
> localized** — because the score averages a whole grid, its evidence is spread
> across patches rather than concentrated — which is a stated limitation and the
> clear next step, not something we paper over. Moving v1 → v2 raised image AUC
> from 0.693 to 0.780 by adding interpretable features while keeping the frozen
> `predict_fractal` interface byte-for-byte identical.
