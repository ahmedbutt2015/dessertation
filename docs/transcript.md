 1 # Infographic / Animated Presentation Script                                                                                                                                                                                                                         
         2 # Explainable Exoplanet Transit Classification                                                                                                                                                                                                                       
         3 # Ahmed Fayyaz Butt — COM748 Masters Dissertation                                                                                                                                                                                                                    
         4                                                                                                                                                                                                                                                                      
         5 ---                                                                                                                                                                                                                                                                  
         6
         7 > **HOW TO USE THIS SCRIPT**
         8 > - Upload to Canva (canva.com) — use "Video Presentation" or "Animated Infographic" template
         9 > - Each SLIDE section = one Canva page/slide
        10 > - VISUAL = what you design/draw on that slide
        11 > - ON-SCREEN TEXT = the text that appears (animate it letter by letter or fade in)
        12 > - NARRATION = voiceover or caption text at the bottom
        13 > - ANIMATION CUE = the transition or motion effect to apply
        14 > - Recommended export: MP4 video at 1920×1080, or animated GIF
        15
        16 ---
        17
        18 ---
        19
        20 # ═══════════════════════════════════════
        21 # ACT 0 — TITLE & HOOK
        22 # ═══════════════════════════════════════
        23
        24 ---
        25
        26 ## SLIDE 01 — COLD OPEN
        27
        28 **VISUAL:**
        29 Full black screen. A single white dot appears in the centre. Slowly, thousands more white dots fade in — a starfield. One star pulses with a very faint rhythmic dimming, almost invisible. A thin orbit line traces around it. A tiny dot crosses in front of
            it.
        30
        31 **ON-SCREEN TEXT (fades in, centre, large font):**
        32 > "Out of 200,000 stars observed by NASA's Kepler telescope…"
        33 > "…only 5,302 showed signs of a planet."
        34 > "How do we know which ones are real?"
        35
        36 **NARRATION:**
        37 "Between 2009 and 2018, NASA's Kepler Space Telescope stared at a single patch of sky without blinking. It watched over 200,000 stars for one thing: a planet passing in front of its star. But identifying genuine planetary signals from noise, instrument e
           rror, and false alarms — that is a problem that still challenges astronomers today."
        38
        39 **ANIMATION CUE:**
        40 Stars fade in slowly (2s). One star pulses dimmer/brighter (loop 3×). Text lines appear one by one with a 0.5s gap each.
        41
        42 ---
        43
        44 ## SLIDE 02 — PROJECT TITLE
        45
        46 **VISUAL:**
        47 Dark navy background with a subtle star texture. A glowing circular GAF image floats on the right (a 64×64 blue/orange gradient ring pattern — this is what a GAF image looks like). On the left: project title in clean white sans-serif.
        48
        49 **ON-SCREEN TEXT:**
        50 > **Explainable Exoplanet Transit Classification**
        51 > Using Vision Transformers, LoRA, and Retrieval-Augmented AI
        52 > on NASA Kepler Light Curves
        53 >
        54 > Ahmed Fayyaz Butt | COM748 Masters Research Project
        55 > Ulster University | Supervisor: Mubashir Ali Cheema
        56
        57 **NARRATION:**
        58 "This project builds an AI system that classifies Kepler planet candidates as genuine planets or false positives — and then explains why, by citing real confirmed planetary systems from NASA's archive."
        59
        60 **ANIMATION CUE:**
        61 GAF image rotates slowly (continuous). Title fades in from left. Subtitle fades in 1s later.
        62
        63 ---
        64
        65 ---
        66
        67 # ═══════════════════════════════════════
        68 # ACT 1 — THE PROBLEM
        69 # ═══════════════════════════════════════
        70
        71 ---
        72
        73 ## SLIDE 03 — WHAT IS THE TRANSIT METHOD?
        74
        75 **VISUAL:**
        76 A large orange star on the left. A small dark planet moves across it from right to left (animated). Below the star: a line graph of brightness over time — flat, then a dip as the planet crosses, then flat again. The dip is labelled "TRANSIT DIP".
        77
        78 **ON-SCREEN TEXT:**
        79 > "The Transit Method"
        80 > When a planet crosses its star, it blocks a tiny fraction of the star's light.
        81 > Kepler measured this brightness drop to detect planets.
        82 > A real planet causes a drop of less than 1% — barely visible.
        83
        84 **NARRATION:**
        85 "The transit method works like this: imagine holding a marble in front of a flashlight. The flashlight dims slightly. Kepler detected exactly this effect — brightness drops of as little as 0.01% — for hundreds of thousands of stars, every 30 minutes, for
            4 years. That generated terabytes of data."
        86
        87 **ANIMATION CUE:**
        88 Planet animates across the star (3s, loop twice). The brightness graph draws itself in real-time as the planet crosses. The dip highlight pulses yellow.
        89
        90 ---
        91
        92 ## SLIDE 04 — THE SCALE OF THE PROBLEM
        93
        94 **VISUAL:**
        95 A large number counter animating upward. Three columns side by side:
        96 - Column 1: "Stars Watched" → 200,000 (counter rolls up)
        97 - Column 2: "Suspicious Signals Found" → 9,564 KOIs (counter rolls up, slower)
        98 - Column 3: "Confirmed Planets So Far" → 3,397 (counter rolls up, slowest)
        99
       100 Below, a pie chart splits: CONFIRMED (green, 36.2%) | FALSE POSITIVE (red, 63.8%) | CANDIDATE (grey).
       101
       102 **ON-SCREEN TEXT:**
       103 > 200,000 stars watched
       104 > 9,564 suspicious signals flagged as KOIs (Kepler Objects of Interest)
       105 > Each KOI must be manually reviewed by astronomers
       106 > Only 36.2% turn out to be genuine planets
       107
       108 **NARRATION:**
       109 "Kepler flagged 9,564 suspicious brightness dips as Kepler Objects of Interest — KOIs. Each one required analysis. NASA astronomers labelled them CONFIRMED, FALSE POSITIVE, or left them as CANDIDATE when uncertain. The false alarm rate is over 60%. Manua
           l vetting at this scale is simply not sustainable — especially as NASA's newer TESS telescope is generating 20,000 new candidates every single year."
       110
       111 **ANIMATION CUE:**
       112 Counters animate upward (2s each, staggered). Pie chart draws itself clockwise. False positive slice flashes red briefly.
       113
       114 ---
       115
       116 ## SLIDE 05 — WHAT IS A FALSE POSITIVE?
       117
       118 **VISUAL:**
       119 Two side-by-side light curve graphs — both showing a dip. Left labelled "REAL PLANET" (clean, symmetric dip). Right labelled "FALSE POSITIVE" (similar dip, but with subtle asymmetry or secondary eclipse). Below each: icons showing the cause.
       120
       121 False positive causes listed as icons:
       122 - Binary star eclipsing each other (two stars icon)
       123 - Background star contamination (overlapping circles)
       124 - Instrument glitch (lightning bolt)
       125
       126 **ON-SCREEN TEXT:**
       127 > "Not every dip is a planet."
       128 > False positives are caused by:
       129 > — Eclipsing binary stars (two stars mimicking a planet)
       130 > — Background star contamination (light from a nearby star)
       131 > — Instrument artefacts (sensor noise)
       132 > Telling them apart from genuine planets is the core challenge.
       133
       134 **NARRATION:**
       135 "A false positive occurs when something other than a planet causes the brightness dip. The most common culprit is an eclipsing binary — two stars orbiting each other, one periodically blocking the other. From Kepler's viewpoint, this looks almost identic
           al to a planet transit. Distinguishing these at scale, accurately, is exactly what machine learning is designed for."
       136
       137 **ANIMATION CUE:**
       138 Both light curves draw themselves simultaneously. False positive curve shakes slightly (asymmetry highlighted in orange). Cause icons pop in one by one with sound-effect-style bounce.
       139
       140 ---
       141
       142 ## SLIDE 06 — WHY THIS IS NOT A TOY PROBLEM
       143
       144 **VISUAL:**
       145 Split screen. Left: an astronomer at a desk manually reviewing plots (illustrated figure). Clock icon showing "~30 minutes per KOI". Right: the TESS satellite with "20,000 new candidates / year" text. A red X through the astronomer figure. Green checkmar
           k on an AI chip icon.
       146
       147 **ON-SCREEN TEXT:**
       148 > Manual vetting: ~30 minutes per KOI
       149 > 9,564 KOIs = ~4,782 hours of human expert time
       150 > TESS is adding ~20,000 new candidates every year
       151 > This is an active, funded NASA research problem — not a classroom exercise.
       152
       153 **NARRATION:**
       154 "This is not an academic toy problem. NASA employs teams of astronomers specifically to vet planet candidates, and the backlog is growing. Automating this classification — and crucially, making the AI decision explainable to scientists who need to trust
           it — is exactly what this project addresses."
       155
       156 **ANIMATION CUE:**
       157 Clock spins fast on left. Counter fills up on right. Red X animates onto astronomer figure. Checkmark pops in on AI chip.
       158
       159 ---
       160
       161 ---
       162
       163 # ═══════════════════════════════════════
       164 # ACT 2 — WHAT EXISTED BEFORE
       165 # ═══════════════════════════════════════
       166
       167 ---
       168
       169 ## SLIDE 07 — THE BASELINE: ASTRONET (2018)
       170
       171 **VISUAL:**
       172 A CNN architecture diagram — classic funnel shape. Input light curve (1D waveform) on the left. Convolutional layers shown as stacked rectangles narrowing down. Final output: "PLANET / NOT PLANET". Paper citation at bottom: "Shallue & Vanderburg, 2018".
       173
       174 **ON-SCREEN TEXT:**
       175 > "AstroNet (2018) — The First Deep Learning Classifier"
       176 > Shallue & Vanderburg applied a Convolutional Neural Network (CNN)
       177 > to phase-folded Kepler light curves.
       178 > Result: ~96% accuracy on the test set.
       179 > This was the state of the art — and it set the baseline.
       180
       181 **NARRATION:**
       182 "In 2018, Shallue and Vanderburg published AstroNet — the first deep learning system to classify Kepler planet candidates at scale. It used a Convolutional Neural Network on phase-folded light curves and achieved around 96% accuracy. Impressive — but it
           had two critical limitations."
       183
       184 **ANIMATION CUE:**
       185 CNN diagram builds left to right with each layer sliding in. Accuracy number counts up to 96%. Paper citation fades in last.
       186
       187 ---
       188
       189 ## SLIDE 08 — THE TWO LIMITATIONS OF CNN APPROACHES
       190
       191 **VISUAL:**
       192 Two large cards side by side, each with a red border.
       193
       194 Card 1 heading: "LIMITATION 1 — LOCAL PATTERNS ONLY"
       195 Shows a CNN "looking at" a small window of the light curve. Arrow pointing to: "Misses global periodic structure — the shape of the full orbit."
       196
       197 Card 2 heading: "LIMITATION 2 — BLACK BOX"
       198 Shows a CNN outputting "PLANET" with a big question mark below it. Text: "No explanation. No reason. Astronomer cannot verify or challenge the decision."
       199
       200 **ON-SCREEN TEXT:**
       201 > CNNs scan small local windows — they miss the global shape.
       202 > CNNs give no explanation for their decision.
       203 > An astronomer cannot say "I trust this" without knowing why the model decided.
       204
       205 **NARRATION:**
       206 "CNNs are excellent at detecting local patterns — edges, textures, small-scale shapes. But a planet transit is a global periodic signal. Its shape across the entire orbital phase matters. Attention mechanisms, which form the backbone of Vision Transforme
           rs, handle exactly this. And crucially: CNNs give no justification for their output. For scientific use, that is unacceptable."
       207
       208 **ANIMATION CUE:**
       209 Both cards slide in from top. Limitation labels flash red. Question mark spins and stays.
       210
       211 ---
       212
       213 ## SLIDE 09 — THE GAP THIS PROJECT FILLS
       214
       215 **VISUAL:**
       216 A timeline on screen with three points marked:
       217 - 2018: AstroNet — CNN, no explanation
       218 - 2025: Choudhary et al. — ViT + GAF, 89.46% recall, still no explanation
       219 - 2026: THIS PROJECT — ViT + LoRA + Agentic RAG, classification + explanation
       220
       221 Arrow grows along the timeline from left to right. The 2026 point glows brighter than the others.
       222
       223 **ON-SCREEN TEXT:**
       224 > "2025: Choudhary et al. applied ViT + GAF to Kepler — best recall to date."
       225 > "But still no explanation layer."
       226 > "This project adds: parameter-efficient fine-tuning (LoRA) + explainable RAG."
       227 > "No existing exoplanet system does this."
       228
       229 **NARRATION:**
       230 "In 2025, Choudhary et al. showed that Vision Transformers with Gramian Angular Fields achieved 89.46% recall on Kepler data — better than any prior CNN. But even this state-of-the-art system outputs only a label. This project picks up exactly where that
            work stopped: it adds a scientifically grounded explanation layer. That is the novel contribution."
       231
       232 **ANIMATION CUE:**
       233 Timeline draws itself left to right. Each point pops in with a dot. 2026 point pulses with a glow. Text fades in below each point.
       234
       235 ---
       236
       237 ---
       238
       239 # ═══════════════════════════════════════
       240 # ACT 3 — THE DATA PIPELINE
       241 # ═══════════════════════════════════════
       242
       243 ---
       244
       245 ## SLIDE 10 — FROM RAW TELESCOPE DATA TO NUMBERS
       246
       247 **VISUAL:**
       248 Four-step horizontal flow diagram with icons:
       249
       250 Step 1: Telescope icon → "Raw FITS file: 70,000 timestamped brightness readings per star, 4 years"
       251 Step 2: Waveform icon → "Light curve: brightness plotted over time"
       252 Step 3: Fold icon → "Phase-folding: all orbital periods stacked on top of each other → 2001 data points"
       253 Step 4: Grid icon → "Phase-folded curve: one clean transit shape, ready for ML"
       254
       255 **ON-SCREEN TEXT:**
       256 > Raw FITS file → 70,000 time-stamped flux readings per star
       257 > Phase-folding collapses 4 years of data into one clean orbit shape
       258 > Result: 2,001 numbers describing the transit profile
       259 > From gigabytes to a single row in a spreadsheet
       260
       261 **NARRATION:**
       262 "Each Kepler star has roughly 70,000 brightness measurements over 4 years. Phase-folding is the key preprocessing step: we use the known orbital period of each KOI to stack every transit on top of each other. All the noise averages out; the transit shape
            becomes crisp. The result is 2,001 numbers — a compact description of the entire transit profile."
       263
       264 **ANIMATION CUE:**
       265 Steps animate in sequence left to right with connecting arrows. Step 3 shows a visual fold animation (wave collapses into a single clean dip).
       266
       267 ---
       268
       269 ## SLIDE 11 — THE MENDELEY DATASET
       270
       271 **VISUAL:**
       272 A large CSV table visual (stylised). Columns: row index, flux_0, flux_1, ... flux_2000, label. Highlight one row. At the bottom right: dataset stats panel.
       273
       274 Stats panel:
       275 - Total KOIs: 5,302
       276 - Each row: 2,001 flux values + 1 label
       277 - CONFIRMED: 2,195 (41.4%) — green bar
       278 - FALSE POSITIVE: 3,107 (58.6%) — red bar
       279 - File size: ~200MB
       280
       281 **ON-SCREEN TEXT:**
       282 > "Macedo & Zalewski 2024 — Mendeley Data (DOI: 10.17632/wctcv34962.3)"
       283 > Pre-processed Kepler light curves — published open dataset
       284 > 5,302 KOIs, each as 2,001 phase-folded flux values
       285 > Labels built in: CONFIRMED = 1 | FALSE POSITIVE = 0
       286 > Why this dataset: same NASA/MAST source data, citable, reproducible, Kaggle-compatible
       287
       288 **NARRATION:**
       289 "The dataset used is from Macedo and Zalewski, published on Mendeley Data in 2024. It contains 5,302 phase-folded Kepler light curves, each reduced to 2,001 flux values, with labels already assigned. The underlying data comes from the same NASA MAST arch
           ive that Kepler researchers use — so this is not a shortcut. It is a reproducible, citable, peer-reviewed preprocessing of the official data."
       290
       291 **ANIMATION CUE:**
       292 Table rows scroll past fast, then slow and highlight one row. Stats bars fill up (green then red). DOI text types itself in.
       293
       294 ---
       295
       296 ## SLIDE 12 — CLASS IMBALANCE AND WHY IT MATTERS
       297
       298 **VISUAL:**
       299 Two large circles, side by side. Left: green, labelled "CONFIRMED 41.4% (2,195)". Right: red, labelled "FALSE POSITIVE 58.6% (3,107)". Below: a scale/balance tipping toward red.
       300
       301 Then a second visual: a confusion matrix mockup. Shows what happens if you just predict everything as FALSE POSITIVE — you get high accuracy but zero planet recall. Label this "The accuracy trap."
       302
       303 **ON-SCREEN TEXT:**
       304 > The dataset is imbalanced — more false positives than confirmed planets.
       305 > A naive model can reach 58% "accuracy" by predicting everything as FALSE POSITIVE.
       306 > That means it misses every single real planet.
       307 > Solution: Weighted cross-entropy loss + evaluate with F1 and AUC-ROC, NOT accuracy.
       308
       309 **NARRATION:**
       310 "Class imbalance is a critical issue here. If a model simply labels everything as FALSE POSITIVE, it gets 58% accuracy — while missing every real planet. This is why raw accuracy is a meaningless metric for this problem. We use F1 score, which penalises
           missed planets, and AUC-ROC, which measures the model's ability to rank true planets above false alarms across all possible thresholds. We also apply weighted cross-entropy loss during training to force the model to take both classes seriously."
       311
       312 **ANIMATION CUE:**
       313 Circles size proportionally. Scale tips toward red. Confusion matrix fades in. "Accuracy Trap" text flashes red. F1/AUC-ROC text glows green.
       314
       315 ---
       316
       317 ## SLIDE 13 — THE GAF TRANSFORMATION: FROM 1D TO 2D
       318
       319 **VISUAL:**
       320 Left side: a 1D line graph (the phase-folded light curve — 2001 points, showing a transit dip).
       321 Centre: a transformation arrow with "GAF" label and "pyts library".
       322 Right side: a 64×64 coloured image (the GAF — a circular gradient pattern, blue in the centre, orange/yellow on the outer ring).
       323
       324 Below the arrow: a small equation panel explaining in plain English what GAF does.
       325
       326 **ON-SCREEN TEXT:**
       327 > Gramian Angular Field (GAF) — converting time series to images
       328 > Step 1: Rescale the 2,001 flux values to [-1, 1]
       329 > Step 2: Convert each value to an angle using arccos
       330 > Step 3: Build a 64×64 matrix where each pixel = cos(angle_i + angle_j)
       331 > Result: a 2D image that encodes the temporal correlations of the light curve
       332 > Why: Vision Transformers are designed for images, not raw number sequences
       333
       334 **NARRATION:**
       335 "Vision Transformers cannot directly process a sequence of 2,001 numbers the way a recurrent network can. They need an image. The Gramian Angular Field transformation solves this elegantly: it converts each flux value to an angle, then builds a matrix wh
           ere each pixel encodes the cosine sum of two timepoints. The resulting 64×64 image visually encodes the entire temporal structure of the transit. Confirmed planets and false positives produce visually distinct patterns — and the ViT learns to distinguish
            them."
       336
       337 **ANIMATION CUE:**
       338 Light curve draws itself (animated line). Arrow appears with GAF label. Image materialises pixel by pixel (left to right, top to bottom). Zoom in on a few distinctive pixels and show their value.
       339
       340 ---
       341
       342 ## SLIDE 14 — CONFIRMED vs FALSE POSITIVE: WHAT THE GAF LOOKS LIKE
       343
       344 **VISUAL:**
       345 Side by side comparison. Four small GAF images on each side (real examples from the dataset if possible, or representative illustrations).
       346
       347 Left column heading: "CONFIRMED PLANET" — images show a distinctive dark circular core pattern.
       348 Right column heading: "FALSE POSITIVE" — images show a different, more diffuse or asymmetric pattern.
       349
       350 Below: "The ViT learns to tell these apart."
       351
       352 **ON-SCREEN TEXT:**
       353 > Confirmed planets → distinct symmetric transit shape in the GAF image
       354 > False positives → asymmetric, secondary eclipse, or flat patterns
       355 > These visual differences are subtle — too subtle for the human eye at scale
       356 > The Vision Transformer's attention mechanism detects them automatically
       357
       358 **NARRATION:**
       359 "Here is the intuition made visual. Confirmed planet transits produce a characteristic symmetric pattern in the GAF image — a clear periodic signal. False positives, driven by eclipsing binaries or instrument noise, produce subtly different patterns: asy
           mmetric arcs, secondary eclipse signatures, or irregular textures. The ViT does not need to be told what to look for — it learns which features predict the label from the training examples."
       360
       361 **ANIMATION CUE:**
       362 Images fade in one by one. Zoom effect on two representative images. Highlight the distinguishing region with a soft glow circle.
       363
       364 ---
       365
       366 ---
       367
       368 # ═══════════════════════════════════════
       369 # ACT 4 — THE MODEL: VISION TRANSFORMER
       370 # ═══════════════════════════════════════
       371
       372 ---
       373
       374 ## SLIDE 15 — WHAT IS A VISION TRANSFORMER (ViT)?
       375
       376 **VISUAL:**
       377 A 64×64 GAF image on the left. It splits into a 4×4 grid of 16×16 patches (animated — grid lines draw themselves in). Each patch gets a number label (1 through 16). These patches then become a sequence of tokens flowing right into a transformer block dia
           gram.
       378
       379 Transformer block shows: Patch Embedding → Positional Encoding → Multi-Head Self-Attention → MLP → Classification Head → CONFIRMED / FALSE POSITIVE.
       380
       381 **ON-SCREEN TEXT:**
       382 > ViT-B/16 — Vision Transformer (Base, 16×16 patches)
       383 > Step 1: Divide the 64×64 GAF image into 16 patches of 16×16 pixels each
       384 > Step 2: Each patch is flattened into a vector (a "token")
       385 > Step 3: Multi-Head Self-Attention asks: "How does each patch relate to every other patch?"
       386 > Step 4: Classification head outputs: CONFIRMED (1) or FALSE POSITIVE (0)
       387 > Pretrained on ImageNet-21k — 86 million parameters
       388
       389 **NARRATION:**
       390 "The Vision Transformer, introduced by Dosovitskiy et al. in 2021, applies the transformer architecture — originally designed for language — to images. The image is divided into fixed-size patches, each treated as a token, just like a word in a sentence.
            The self-attention mechanism then computes how every patch relates to every other patch globally. This is the key advantage over CNNs: a CNN only looks at local neighbourhoods, but the ViT can directly model long-range dependencies across the entire ima
           ge — which is exactly what a periodic transit signal requires."
       391
       392 **ANIMATION CUE:**
       393 Grid lines draw over image (animated). Patches numbered 1-16. Tokens flow right like a conveyor belt. Attention lines criss-cross (web of connections between patches). Output label flashes green (CONFIRMED) or red (FALSE POSITIVE).
       394
       395 ---
       396
       397 ## SLIDE 16 — WHY ViT BEATS CNN FOR THIS PROBLEM
       398
       399 **VISUAL:**
       400 Two-column comparison table with clean icons.
       401
       402 | Feature | CNN | ViT |
       403 |---|---|---|
       404 | Looks at | Local patches only | Every patch vs every other patch |
       405 | Best for | Textures, edges | Global patterns, periodic signals |
       406 | Transit shape awareness | Partial | Full orbital context |
       407 | Pretrained knowledge | ImageNet | ImageNet (same advantage) |
       408 | Choudhary 2025 recall | — | 89.46% |
       409
       410 **ON-SCREEN TEXT:**
       411 > CNNs process local windows — they can miss that the transit shape extends across the whole image.
       412 > ViT self-attention processes ALL patches simultaneously — it sees the full orbital context.
       413 > Periodic signals like planet transits are GLOBAL patterns.
       414 > ViT is architecturally better suited to this task.
       415
       416 **NARRATION:**
       417 "The comparison is straightforward. A CNN applies filters to small local regions and builds up understanding from the bottom up. A ViT, through self-attention, directly models how every part of the GAF image relates to every other part. For a transit sig
           nal — which is a global periodic pattern embedded in the image — the ViT's global attention is the right tool. Choudhary et al. confirmed this empirically in 2025 with 89.46% recall."
       418
       419 **ANIMATION CUE:**
       420 Table rows slide in one by one. CNN column cells highlighted in amber. ViT column cells highlighted in green. Final row (recall) counter animates to 89.46%.
       421
       422 ---
       423
       424 ---
       425
       426 # ═══════════════════════════════════════
       427 # ACT 5 — LoRA: EFFICIENT FINE-TUNING
       428 # ═══════════════════════════════════════
       429
       430 ---
       431
       432 ## SLIDE 17 — THE FINE-TUNING PROBLEM
       433
       434 **VISUAL:**
       435 A massive weight matrix visual — 86 million dots arranged in a grid. Label: "ViT-B/16 — 86,000,000 parameters". Below it: "Your training data — 3,711 examples". Arrow pointing from training data to the weight matrix with a red warning sign: "OVERFITTING
           RISK. CATASTROPHIC FORGETTING RISK."
       436
       437 Then a clock: "Full fine-tuning time on GPU: ~8 hours per run". Red.
       438
       439 **ON-SCREEN TEXT:**
       440 > ViT-B/16 has 86 million parameters.
       441 > Our training set has only 3,711 labelled examples.
       442 > Full fine-tuning: update all 86M parameters on 3,711 examples.
       443 > Risk: catastrophic forgetting (model loses its pretrained knowledge)
       444 > Risk: overfitting (memorises training data, fails on new data)
       445 > Risk: 8+ hours of GPU training per experiment
       446
       447 **NARRATION:**
       448 "The pretrained ViT-B/16 already knows how to interpret images — it has been trained on 14 million images. Full fine-tuning would update every one of its 86 million parameters on our 3,711 training examples. With this data-to-parameter ratio, the model w
           ould likely overfit or forget its pretrained visual knowledge. We need a smarter approach."
       449
       450 **ANIMATION CUE:**
       451 86M dots fill the screen (fast). 3,711 dots appear below (much smaller cluster). Warning icon pulses. Clock spins fast and lands on "8 hours" in red.
       452
       453 ---
       454
       455 ## SLIDE 18 — WHAT IS LoRA?
       456
       457 **VISUAL:**
       458 A weight matrix W (large, greyed out — frozen). Two small matrices A and B appear beside it, coloured bright blue. Arrow from A → B showing multiplication. The result ΔW is added back to W with a + sign.
       459
       460 Formula on screen: W' = W + ΔW = W + A × B
       461
       462 Label on W: "Original weights — FROZEN, never updated"
       463 Label on A × B: "LoRA adapters — ~600K parameters — TRAINABLE"
       464
       465 **ON-SCREEN TEXT:**
       466 > LoRA — Low-Rank Adaptation (Hu et al. 2022)
       467 > Key idea: Don't update the original weights.
       468 > Instead, insert two tiny trainable matrices (A and B) beside each attention layer.
       469 > Their product (A × B) is a low-rank approximation of the weight update.
       470 > Only ~600,000 parameters are trained instead of 86,000,000.
       471 > That is less than 1% of the total model.
       472
       473 **NARRATION:**
       474 "LoRA — Low-Rank Adaptation — is a parameter-efficient fine-tuning technique introduced by Hu et al. in 2022. The insight is elegant: instead of updating the original weight matrix W, freeze it completely and insert two small matrices — A and B — whose p
           roduct represents the adaptation. The rank r controls how many parameters are in A and B. At rank 8, this means roughly 600,000 trainable parameters instead of 86 million. The pretrained knowledge is preserved; only the task-specific adaptation is learne
           d."
       475
       476 **ANIMATION CUE:**
       477 Large W matrix greys out with a padlock icon. A and B matrices slide in from the side (blue, glowing). Multiplication arrow animates. + sign pops in. W' lights up.
       478
       479 ---
       480
       481 ## SLIDE 19 — WHY LoRA IS RIGHT FOR THIS PROBLEM
       482
       483 **VISUAL:**
       484 Three-column layout showing why each concern is resolved:
       485
       486 Column 1: "Only 3,711 training examples"
       487 → LoRA's 600K params: right-sized for this data. Full fine-tuning's 86M params: catastrophic overfit.
       488
       489 Column 2: "GPU budget is limited (30 hrs/week free)"
       490 → LoRA trains in ~2 hours. Full fine-tuning: 8+ hours. Saves 4× GPU time.
       491
       492 Column 3: "We need to compare settings fairly"
       493 → LoRA rank r is a clean research variable: r=4 vs r=8 vs r=16 = three rigorous experiments.
       494
       495 **ON-SCREEN TEXT:**
       496 > LoRA solves all three constraints:
       497 > 1. Data-to-parameter ratio: 600K parameters for 3,711 examples is appropriate
       498 > 2. GPU efficiency: ~2 hours training vs 8+ for full fine-tuning
       499 > 3. Research variable: rank r=4, r=8, r=16 gives three comparable experiments
       500 > This is not a shortcut — it is the state-of-the-art approach for low-data regimes.
       501
       502 **NARRATION:**
       503 "LoRA is not being used here because it is trendy. It is being used because the constraints of this problem demand it. Small dataset, limited GPU budget, and the need for a clean experimental variable all point to the same solution. Rank r is the hyperpa
           rameter being studied — r=4 gives ~150K new parameters, r=8 gives ~600K, r=16 gives ~1.2M. The comparison of these three settings is a direct research contribution."
       504
       505 **ANIMATION CUE:**
       506 Three columns slide in from bottom. Each concern shows as a red problem that a green LoRA solution resolves (animated X → checkmark transition).
       507
       508 ---
       509
       510 ## SLIDE 20 — THE FOUR EVALUATION PARADIGMS
       511
       512 **VISUAL:**
       513 A staircase diagram going upward from left to right. Each step labelled and coloured differently:
       514
       515 Step 1 (grey): ZERO-SHOT — "No adaptation. Raw pretrained ViT on GAF images."
       516 Step 2 (yellow): ONE-SHOT — "1 labelled example per class. Nearest-neighbour style adaptation."
       517 Step 3 (orange): FEW-SHOT — "Small set of labelled examples. Prototypical classification."
       518 Step 4 (green): LoRA — "Full LoRA fine-tuning. r=4, r=8, r=16 variants."
       519
       520 Arrow along the staircase: "Increasing adaptation → increasing performance expected"
       521
       522 **ON-SCREEN TEXT:**
       523 > This is not just LoRA vs no LoRA.
       524 > It is a systematic study of how much adaptation a ViT needs for this task.
       525 > Zero-shot: what does a pretrained ImageNet ViT already know about transits?
       526 > One-shot: can a single example unlock classification?
       527 > Few-shot: how quickly does performance scale with examples?
       528 > LoRA: what is the ceiling of parameter-efficient fine-tuning here?
       529
       530 **NARRATION:**
       531 "This project is not simply testing whether LoRA works. It is answering a deeper question: how much task-specific adaptation does a Vision Transformer actually need to classify exoplanet transits? Each paradigm is a valid, publishable experiment in itsel
           f. The zero-shot result tells us what pretrained visual knowledge already transfers. The progression to LoRA maps out the learning efficiency curve of the ViT architecture on this specific scientific domain."
       532
       533 **ANIMATION CUE:**
       534 Staircase builds step by step from left to right. Each step's label types in. Arrow grows along the staircase. Performance bar beside each step fills up progressively (zero-shot lowest, LoRA highest).
       535
       536 ---
       537
       538 ---
       539
       540 # ═══════════════════════════════════════
       541 # ACT 6 — THE CLASSIFICATION PIPELINE (END TO END)
       542 # ═══════════════════════════════════════
       543
       544 ---
       545
       546 ## SLIDE 21 — THE FULL PIPELINE (OVERVIEW)
       547
       548 **VISUAL:**
       549 A clean horizontal pipeline diagram, left to right, with coloured boxes connected by arrows:
       550
       551 [RAW FITS / Mendeley CSV] → [Phase-fold + normalize] → [GAF 64×64 image] → [ViT-B/16 + LoRA] → [CONFIRMED / FALSE POSITIVE + Confidence Score]
       552
       553 Each box has an icon above it. The pipeline is animated — a small "particle" (bright dot) flows along the arrows from left to right.
       554
       555 **ON-SCREEN TEXT:**
       556 > The Classification Pipeline:
       557 > 1. Input: phase-folded Kepler light curve (2,001 flux values)
       558 > 2. Rescale to [-1, 1]
       559 > 3. GAF transformation → 64×64 image
       560 > 4. ViT-B/16 with LoRA adapters → classification
       561 > 5. Output: CONFIRMED (1) or FALSE POSITIVE (0) + probability score
       562
       563 **NARRATION:**
       564 "Here is the classification pipeline in full. A light curve enters on the left. It is rescaled, transformed into a GAF image, and fed into the ViT-B/16 model with its LoRA adapters. The model outputs a probability — the higher it is, the more confident t
           he model is that this is a genuine planet. This pipeline runs in milliseconds on GPU. The bottleneck is not speed — it is trust. Which is where the next module comes in."
       565
       566 **ANIMATION CUE:**
       567 Pipeline draws left to right with arrows. Particle animation flows through it (loops twice). Each box pulses as the particle passes through it.
       568
       569 ---
       570
       571 ---
       572
       573 # ═══════════════════════════════════════
       574 # ACT 7 — RAG: THE EXPLANATION LAYER
       575 # ═══════════════════════════════════════
       576
       577 ---
       578
       579 ## SLIDE 22 — THE TRUST PROBLEM
       580
       581 **VISUAL:**
       582 An astronomer (illustrated figure) sitting at a computer. The screen shows: "RESULT: CONFIRMED. Confidence: 87%." The astronomer has a speech bubble: "...But WHY?"
       583
       584 Below: a telescope (Kepler) with a question mark over it. Next to it: a paper titled "Journal of Astronomy" with a rejection stamp — "INSUFFICIENT JUSTIFICATION."
       585
       586 **ON-SCREEN TEXT:**
       587 > A classification label is not enough for science.
       588 > No peer-reviewed journal accepts: "The AI said it's a planet."
       589 > Astronomers need: comparable systems, physical parameters, evidence.
       590 > The question is not just WHAT — it is WHY.
       591
       592 **NARRATION:**
       593 "Imagine presenting your results to a review board or submitting to a journal. The ViT says CONFIRMED with 87% confidence. An astronomer's first question is: 'Compared to what? What known systems does this resemble? What are the orbital mechanics?' A lab
           el without evidence is not science. This is the gap that the Retrieval-Augmented Generation module closes."
       594
       595 **ANIMATION CUE:**
       596 Astronomer figure has a question mark appearing above their head. Confidence score types itself in. Rejection stamp slams down on paper. Question mark pulses.
       597
       598 ---
       599
       600 ## SLIDE 23 — WHAT IS RAG?
       601
       602 **VISUAL:**
       603 Three-step visual:
       604
       605 Step 1: "QUERY" — the classified KOI with its physical parameters (period, depth, radius, stellar temp) represented as a vector: [0.23, 0.45, 0.12, 0.67, ...]
       606
       607 Step 2: "SEARCH" — this vector compared against a database of 6,128 confirmed planet vectors in a FAISS index. Five vectors light up as "nearest neighbours."
       608
       609 Step 3: "RETRIEVE" — five confirmed planet profiles shown as cards: "Kepler-452b", "Kepler-186f", etc. with their parameters.
       610
       611 **ON-SCREEN TEXT:**
       612 > RAG — Retrieval-Augmented Generation
       613 > After classification: represent the KOI as a vector of physical parameters
       614 > Search NASA Exoplanet Archive: 6,128 confirmed planetary systems
       615 > Retrieve the 5 most physically similar confirmed planets (by cosine similarity)
       616 > These 5 become the evidence base for the explanation
       617
       618 **NARRATION:**
       619 "Retrieval-Augmented Generation works by first converting the candidate KOI's physical properties into a vector — a numerical fingerprint representing its orbital period, transit depth, planet radius, stellar temperature, and other parameters. This vecto
           r is then compared against a pre-built database of 6,128 confirmed exoplanets from the NASA Exoplanet Archive. The 5 closest matches — the most physically similar confirmed systems — are retrieved. These are not randomly chosen examples. They are the rea
           l, peer-reviewed confirmed planets that most closely resemble this candidate."
       620
       621 **ANIMATION CUE:**
       622 Vector appears as glowing numbers. FAISS database shown as a grid of dots — 5 light up (highlighted). Five cards slide out. Cosine similarity score displayed on each card.
       623
       624 ---
       625
       626 ## SLIDE 24 — THE FAISS INDEX
       627
       628 **VISUAL:**
       629 A visual of the NASA Exoplanet Archive as a "knowledge vault" — a library icon with "6,128 confirmed systems" on the door. Inside: rows of "system profiles" as file cards. A search beam (laser line) enters the vault and picks out 5 cards.
       630
       631 Below: a simple technical note showing what each row in the index contains:
       632 - System name (e.g. "Kepler-452b")
       633 - Orbital period (days)
       634 - Transit depth (ppm)
       635 - Planet radius (Earth radii)
       636 - Stellar effective temperature (K)
       637 - Stellar radius (Solar radii)
       638
       639 **ON-SCREEN TEXT:**
       640 > The Knowledge Base: NASA Exoplanet Archive
       641 > 6,128 confirmed planetary systems — peer-reviewed, verified
       642 > Each system stored as a vector of orbital + stellar parameters
       643 > FAISS (Facebook AI Similarity Search) — finds the 5 nearest vectors in milliseconds
       644 > This is not an LLM hallucinating facts — it is direct retrieval from a verified scientific database
       645
       646 **NARRATION:**
       647 "The FAISS index is the knowledge vault of the system. FAISS — Facebook AI Similarity Search — is a library designed to search millions of vectors in milliseconds. Every row in our index corresponds to a real, peer-reviewed confirmed planet from NASA's E
           xoplanet Archive. When the system retrieves 5 similar systems, it is not generating facts from a language model's memory — it is performing an exact lookup against verified scientific data. This is a critical distinction for scientific credibility."
       648
       649 **ANIMATION CUE:**
       650 Library vault door opens (animated). Search beam scans across cards. 5 cards pull out and fan open. FAISS logo appears. "NOT hallucination — RETRIEVAL" appears with green checkmark.
       651
       652 ---
       653
       654 ## SLIDE 25 — FROM RETRIEVAL TO EXPLANATION
       655
       656 **VISUAL:**
       657 Input at top: "Candidate KOI-7016.01 — Classified: CONFIRMED (91% confidence)"
       658 Below: 5 retrieved similar systems as cards (with their key parameters)
       659 Arrow pointing down to: a formatted explanation text block.
       660
       661 The explanation text reads (example):
       662 ---
       663 "KOI-7016.01 shows a transit depth of 312 ppm, an orbital period of 8.4 days, and a planet radius of 1.8 Earth radii, orbiting a Sun-like star (Teff 5,840 K).
       664
       665 The 5 most similar confirmed systems are: Kepler-452b (period 384.8d, depth 700ppm), Kepler-186f (period 129.9d, depth 350ppm), Kepler-442b (period 112.3d, depth 2800ppm), Kepler-62f (period 267.3d, depth 480ppm), and Kepler-296e (period 34.1d, depth 138
           0ppm).
       666
       667 These systems share compatible stellar temperatures (5,200–6,100 K) and sub-Neptune radii. The classification confidence of 91% is consistent with this dense cluster of confirmed analogues."
       668 ---
       669
       670 **ON-SCREEN TEXT:**
       671 > The RAG explanation anchors the AI decision in real science:
       672 > — Cites specific confirmed planets by name
       673 > — Compares physical parameters quantitatively
       674 > — Explains why the confidence level is justified
       675 > — Gives astronomers a starting point for independent verification
       676
       677 **NARRATION:**
       678 "Here is the explanation output for a hypothetical candidate. The model's decision is now grounded in five real, named planetary systems. An astronomer reading this report can immediately evaluate whether the analogues are reasonable — are the orbital pe
           riods compatible? Does the stellar temperature match? Is the transit depth in the right range? The AI has done the retrieval work; the scientist can now do what scientists do: exercise judgment."
       679
       680 **ANIMATION CUE:**
       681 KOI card drops in from top. 5 retrieved cards fan in below it. Arrow grows down. Explanation text types itself in (typewriter effect). Key numbers (depths, periods) highlight in gold as they appear.
       682
       683 ---
       684
       685 ---
       686
       687 # ═══════════════════════════════════════
       688 # ACT 8 — AGENTIC RAG: THE LANGGRAPH PIPELINE
       689 # ═══════════════════════════════════════
       690
       691 ---
       692
       693 ## SLIDE 26 — STATIC RAG vs AGENTIC RAG
       694
       695 **VISUAL:**
       696 Two flow diagrams side by side.
       697
       698 LEFT — "Static RAG (basic)":
       699 Input → Classify → Retrieve → Generate Explanation → Output
       700 (Linear, no feedback, no decisions)
       701
       702 RIGHT — "Agentic RAG (this project)":
       703 Input → [AGENT] → Classify? → Retrieve? → "Are retrieved cases relevant?" → If yes: Generate → Output. If no: Flag low confidence → Adjusted output with caveat.
       704
       705 The right diagram shows branching, decision points, and feedback loops. The agent is shown as a brain icon at the centre.
       706
       707 **ON-SCREEN TEXT:**
       708 > Static RAG: always retrieves, always generates — no quality check
       709 > Agentic RAG: the agent DECIDES whether to retrieve, evaluates quality, flags uncertainty
       710 > If the retrieved systems are not actually similar: say so, rather than generating a misleading explanation
       711 > This is the difference between automation and intelligence
       712
       713 **NARRATION:**
       714 "A static RAG pipeline always retrieves, always generates — it has no awareness of whether the retrieval was any good. An agentic system, built with LangGraph, can reason about its own outputs. If the retrieved systems are dissimilar to the candidate — s
           ay the closest match has only 40% cosine similarity — the agent flags this: 'Retrieval confidence is low. No strong confirmed analogue found. Treat this classification with caution.' That kind of self-aware uncertainty is what makes the output scientific
           ally useful rather than misleadingly confident."
       715
       716 **ANIMATION CUE:**
       717 Left diagram builds linearly (boring, grey). Right diagram builds with branching arrows, agent icon at centre pulsing, decision diamonds in yellow. Comparison contrast is immediate and visual.
       718
       719 ---
       720
       721 ## SLIDE 27 — THE THREE TOOLS OF THE LANGGRAPH AGENT
       722
       723 **VISUAL:**
       724 A central "AGENT" node (hexagon, glowing). Three tool nodes connected by arrows:
       725
       726 Tool 1 (blue): "CLASSIFIER TOOL"
       727 → Runs ViT-B/16 + LoRA on the GAF image
       728 → Returns: label (CONFIRMED/FP) + confidence score
       729
       730 Tool 2 (orange): "RETRIEVER TOOL"
       731 → Queries FAISS index with KOI physical parameters
       732 → Returns: k=5 most similar confirmed systems + similarity scores
       733
       734 Tool 3 (green): "EXPLAINER TOOL"
       735 → Takes classifier output + retrieved systems
       736 → Calls Claude API / Llama-3 / template fallback
       737 → Returns: structured natural language explanation
       738
       739 **ON-SCREEN TEXT:**
       740 > The LangGraph agent has three tools:
       741 > Tool 1 — Classifier: ViT-B/16 + LoRA → label + confidence
       742 > Tool 2 — Retriever: FAISS search → 5 analogues + similarity scores
       743 > Tool 3 — Explainer: LLM/template → natural language report
       744 > The agent orchestrates these tools. It can call them in sequence, in parallel, or conditionally.
       745 > LangGraph provides the state machine that manages this workflow.
       746
       747 **NARRATION:**
       748 "LangGraph is a framework for building agentic AI workflows as stateful graphs. The agent has three tools available. It calls the classifier first, then the retriever. Before calling the explainer, it evaluates the retrieval quality. If similarity scores
            are too low, it modifies the prompt to the explainer to reflect uncertainty. The LLM used for explanation has a priority fallback: Claude API → Ollama with Llama-3 → a structured template. This means the system works even without API access."
       749
       750 **ANIMATION CUE:**
       751 Agent hexagon pulses at centre. Three tool nodes animate in with connecting arrows. Each tool activates in sequence (blue lights up → orange lights up → green lights up). Arrows animate flowing back to agent.
       752
       753 ---
       754
       755 ---
       756
       757 # ═══════════════════════════════════════
       758 # ACT 9 — WEBSITE DEMO (MOCK UI)
       759 # ═══════════════════════════════════════
       760
       761 ---
       762
       763 ## SLIDE 28 — THE RESEARCHER INTERFACE (INTRODUCTION)
       764
       765 **VISUAL:**
       766 A browser window mockup. Clean, modern design. Dark navy header with title: "ExoExplain — Kepler KOI Classification & Explanation System". Tagline below: "Powered by ViT-B/16 + LoRA + Agentic RAG | NASA Kepler Data"
       767
       768 Navigation bar: [Classify] [About] [Model Performance] [API Docs]
       769
       770 A simple input panel centred on the page:
       771
       772 ```
       773 ┌─────────────────────────────────────────────────────────────┐
       774 │   Enter KOI Name or ID:   [ K07016.01          ] [ANALYSE]  │
       775 │   — or —                                                     │
       776 │   Upload light curve CSV:  [ Choose File ]                   │
       777 │                                                              │
       778 │   LoRA Model:  ○ r=4   ● r=8 (recommended)   ○ r=16         │
       779 │   Retrieval k: ○ k=3   ● k=5 (recommended)   ○ k=10         │
       780 └─────────────────────────────────────────────────────────────┘
       781 ```
       782
       783 **ON-SCREEN TEXT:**
       784 > The researcher types in a KOI name (e.g. K07016.01)
       785 > Or uploads their own phase-folded light curve
       786 > Selects the LoRA rank and retrieval k
       787 > Clicks ANALYSE
       788
       789 **NARRATION:**
       790 "To make this system accessible to researchers and educators, the project includes a web interface prototype. A researcher enters a Kepler Object of Interest name — for example K07016.01 — or uploads their own light curve file. They select which model co
           nfiguration to use. They click Analyse. The system processes the request through the full pipeline and returns a structured report."
       791
       792 **ANIMATION CUE:**
       793 Browser window fades in. Cursor clicks into KOI input field. Text types in: "K07016.01". Cursor clicks ANALYSE button. Loading spinner begins.
       794
       795 ---
       796
       797 ## SLIDE 29 — PROCESSING ANIMATION
       798
       799 **VISUAL:**
       800 The browser window now shows a progress panel replacing the input form. Four steps with animated progress indicators:
       801
       802 ```
       803 ┌─────────────────────────────────────────────────────────────┐
       804 │   Processing K07016.01...                                    │
       805 │                                                              │
       806 │   ✅  Step 1: Loading light curve from Mendeley index       │
       807 │   ✅  Step 2: GAF transformation (64×64 image generated)    │
       808 │   ⏳  Step 3: ViT-B/16 + LoRA classification... (running)   │
       809 │   ⬜  Step 4: FAISS retrieval + explanation generation       │
       810 │                                                              │
       811 │   [████████████████░░░░░░░░░░░░]  65%                       │
       812 └─────────────────────────────────────────────────────────────┘
       813 ```
       814
       815 **ON-SCREEN TEXT:**
       816 > Each step runs in sequence under the agent's control.
       817 > The GAF image is generated in real time.
       818 > The ViT runs inference in under 1 second on GPU.
       819 > FAISS retrieval across 6,128 systems takes milliseconds.
       820 > Total time from click to result: under 5 seconds.
       821
       822 **NARRATION:**
       823 "While the system processes, the interface shows each pipeline step completing in real time. The user can see that their input was received, the GAF image was generated, the classifier is running, and retrieval is next. This transparency builds trust — t
           he researcher is not waiting for a black box."
       824
       825 **ANIMATION CUE:**
       826 Step 1 checkbox fills with ✅ (animated). Step 2 same. Step 3 spinner animates. Progress bar fills. Then Step 3 completes, Step 4 begins.
       827
       828 ---
       829
       830 ## SLIDE 30 — THE RESULT PAGE (CLASSIFICATION RESULT)
       831
       832 **VISUAL:**
       833 Full results page in the browser. Clean layout with three panels.
       834
       835 PANEL 1 — TOP (Classification Banner):
       836 ```
       837 ┌─────────────────────────────────────────────────────────────┐
       838 │                                                              │
       839 │   KOI K07016.01                                              │
       840 │                                                              │
       841 │   🟢  CONFIRMED PLANET                                       │
       842 │       Confidence: 91.3%                                      │
       843 │       LoRA model r=8 | AUC-ROC (test): 0.941                 │
       844 │                                                              │
       845 │   "The ViT-B/16 model with LoRA (r=8) classifies this KOI   │
       846 │    as a confirmed planet with high confidence."              │
       847 └─────────────────────────────────────────────────────────────┘
       848 ```
       849
       850 PANEL 2 — MIDDLE LEFT (GAF Image display):
       851 The 64×64 GAF image rendered on screen with caption: "Gramian Angular Field — input to classifier"
       852
       853 PANEL 2 — MIDDLE RIGHT (KOI Parameters):
       854 ```
       855   Orbital Period:    8.41 days
       856   Transit Depth:     312 ppm
       857   Planet Radius:     1.82 Earth radii
       858   Stellar Temp:      5,840 K
       859   Stellar Radius:    1.02 Solar radii
       860   Transit Duration:  2.7 hours
       861 ```
       862
       863 **ON-SCREEN TEXT:**
       864 > The researcher immediately sees: CONFIRMED PLANET, 91.3% confidence
       865 > The GAF image that the model actually classified is displayed
       866 > The physical parameters used for retrieval are shown
       867 > No ambiguity about what went into the decision
       868
       869 **NARRATION:**
       870 "The result panel shows the classification immediately and prominently. Green for confirmed, red for false positive. The confidence score is shown alongside the model performance metric on the held-out test set — so the researcher can calibrate their tru
           st in the score. The GAF image that the classifier actually processed is shown — the researcher can see exactly what the model saw."
       871
       872 **ANIMATION CUE:**
       873 Panel 1 banner fades in with a green flash (CONFIRMED). Confidence number counts up to 91.3%. GAF image renders in. Parameter values count up from zero.
       874
       875 ---
       876
       877 ## SLIDE 31 — THE RESULT PAGE (RAG EXPLANATION)
       878
       879 **VISUAL:**
       880 Panel 3 of the results page — the full explanation section.
       881
       882 ```
       883 ┌─────────────────────────────────────────────────────────────┐
       884 │  EXPLANATION — 5 MOST SIMILAR CONFIRMED SYSTEMS             │
       885 │  (Retrieved from NASA Exoplanet Archive via FAISS)          │
       886 │                                                              │
       887 │  Retrieval Confidence: 0.847 / 1.0  ●●●●○  HIGH            │
       888 │                                                              │
       889 │  #1  Kepler-442b     Period: 112.3d  Depth: 2,800ppm        │
       890 │      Radius: 1.34 RE   Teff: 4,402K   Similarity: 0.94     │
       891 │                                                              │
       892 │  #2  Kepler-296e     Period: 34.1d   Depth: 1,380ppm        │
       893 │      Radius: 1.53 RE   Teff: 4,294K   Similarity: 0.89     │
       894 │                                                              │
       895 │  #3  Kepler-186f     Period: 129.9d  Depth: 350ppm          │
       896 │      Radius: 1.17 RE   Teff: 3,788K   Similarity: 0.86     │
       897 │                                                              │
       898 │  #4  Kepler-62f      Period: 267.3d  Depth: 480ppm          │
       899 │      Radius: 1.41 RE   Teff: 4,925K   Similarity: 0.83     │
       900 │                                                              │
       901 │  #5  Kepler-452b     Period: 384.8d  Depth: 700ppm          │
       902 │      Radius: 1.63 RE   Teff: 5,757K   Similarity: 0.79     │
       903 │                                                              │
       904 └─────────────────────────────────────────────────────────────┘
       905 ```
       906
       907 **ON-SCREEN TEXT:**
       908 > Five real, named, peer-reviewed confirmed planets are retrieved.
       909 > Each one is more similar to this KOI than any other confirmed system in the archive.
       910 > The similarity score is shown so the researcher can evaluate the strength of the analogy.
       911 > Every entry links to the NASA Exoplanet Archive page for that system.
       912 > This is verifiable, traceable evidence — not a generated claim.
       913
       914 **NARRATION:**
       915 "Below the classification, the explanation panel shows the five most similar confirmed planetary systems retrieved from NASA's archive. Each system is named, linked to its NASA page, and displays the similarity score so the researcher can judge how stron
           g the analogy is. A similarity of 0.94 for Kepler-442b means this candidate is nearly identical in parameter space to a system that astronomers have already confirmed with follow-up spectroscopy and radial velocity measurements."
       916
       917 **ANIMATION CUE:**
       918 Panel 3 slides up from bottom. Each retrieved system card slides in with a slight delay (cascade effect). Similarity bars fill from left. Each system name is a clickable hyperlink (cursor hover animation).
       919
       920 ---
       921
       922 ## SLIDE 32 — THE RESULT PAGE (NATURAL LANGUAGE EXPLANATION)
       923
       924 **VISUAL:**
       925 A clean text box below the retrieved systems table. Typewriter animation as the explanation generates.
       926
       927 ```
       928 ┌─────────────────────────────────────────────────────────────┐
       929 │  GENERATED EXPLANATION                           [Copy] [PDF]│
       930 │                                                              │
       931 │  KOI K07016.01 exhibits a transit depth of 312 ppm with an  │
       932 │  orbital period of 8.41 days, consistent with a sub-Earth   │
       933 │  to super-Earth radius planet (1.82 RE) orbiting a Sun-like  │
       934 │  host star (Teff 5,840 K, R★ 1.02 R☉).                      │
       935 │                                                              │
       936 │  The ViT-B/16 classifier (LoRA r=8) assigns this candidate  │
       937 │  a CONFIRMED probability of 91.3%, placing it well above    │
       938 │  the 0.5 decision threshold.                                 │
       939 │                                                              │
       940 │  The 5 nearest confirmed analogues retrieved from the NASA   │
       941 │  Exoplanet Archive (retrieval confidence: 0.847) share       │
       942 │  comparable sub-Neptune radii (1.17–1.63 RE), cool-to-Solar  │
       943 │  host stars (3,788–5,757 K), and transit depths in the       │
       944 │  100–2,800 ppm range.                                        │
       945 │                                                              │
       946 │  The closest analogue, Kepler-442b (similarity: 0.94),      │
       947 │  is a confirmed rocky super-Earth in the habitable zone      │
       948 │  of a K-dwarf star — independently verified by radial       │
       949 │  velocity measurements (Torres et al. 2015).                 │
       950 │                                                              │
       951 │  Assessment: The classifier confidence and retrieval         │
       952 │  consistency jointly support the CONFIRMED classification.   │
       953 │  Recommended for follow-up spectroscopic observation.        │
       954 └─────────────────────────────────────────────────────────────┘
       955 ```
       956
       957 **ON-SCREEN TEXT:**
       958 > The natural language explanation:
       959 > — Summarises the KOI's physical parameters in scientific terms
       960 > — States the model's confidence and what it means
       961 > — Connects the candidate to real confirmed systems by name
       962 > — Cites the peer-reviewed paper that confirmed the closest analogue
       963 > — Gives an actionable recommendation to the astronomer
       964
       965 **NARRATION:**
       966 "The natural language explanation is the final output of the LangGraph agent. It is not a generic template filled in with numbers. It synthesises the classification result, the physical parameters, and the retrieved analogues into a coherent scientific a
           ssessment. The astronomer now has: a confidence score, a retrieval quality score, five named comparable systems, a citation to the most relevant paper, and a recommendation. They can take this report to a colleague, present it at a meeting, or include it
            in a paper. The AI has done the retrieval and synthesis; the scientist makes the final call."
       967
       968 **ANIMATION CUE:**
       969 Text types in with a typewriter cursor effect (key decision phrases highlight in gold as they appear). [Copy] and [PDF] buttons pulse gently at the end. "Recommended for follow-up..." line glows green.
       970
       971 ---
       972
       973 ## SLIDE 33 — LOW-CONFIDENCE CASE (WHAT HAPPENS WHEN THE AGENT IS UNCERTAIN)
       974
       975 **VISUAL:**
       976 A different result panel. This time the classification banner is amber (not red or green):
       977
       978 ```
       979 ┌─────────────────────────────────────────────────────────────┐
       980 │                                                              │
       981 │   KOI K03678.02                                              │
       982 │                                                              │
       983 │   🟡  FALSE POSITIVE                                         │
       984 │       Confidence: 61.2%   ⚠️  LOW CONFIDENCE                 │
       985 │       Retrieval Confidence: 0.43 / 1.0  ●●○○○  WEAK        │
       986 │                                                              │
       987 │   "The model's confidence is below the reliable threshold.  │
       988 │    Retrieved analogues are weakly matched. Manual review    │
       989 │    by a domain expert is strongly recommended before        │
       990 │    acting on this classification."                           │
       991 │                                                              │
       992 └─────────────────────────────────────────────────────────────┘
       993 ```
       994
       995 **ON-SCREEN TEXT:**
       996 > The agentic system knows when it doesn't know.
       997 > Low classifier confidence + low retrieval similarity = honest uncertainty.
       998 > A static pipeline would still generate a confident-sounding explanation.
       999 > The agent flags the case instead — protecting the researcher from misleading outputs.
      1000 > This is what separates an agentic system from a simple predict-and-generate pipeline.
      1001
      1002 **NARRATION:**
      1003 "This is arguably the most important slide in the presentation. When the agent is uncertain, it says so. A 61% confidence score with a 0.43 retrieval match means the system is guessing — and it knows it. Rather than generating a confident-sounding explan
           ation that could mislead an astronomer into treating a false positive as a planet candidate, the agent flags the case for manual expert review. This is responsible AI design. The system is more useful precisely because it acknowledges its own limitations
           ."
      1004
      1005 **ANIMATION CUE:**
      1006 Amber banner flashes in (not green). Warning icon pulses. Low retrieval bar shown (only 2 of 5 dots filled). Recommendation text types in. Contrast this visually with the previous green slide.
      1007
      1008 ---
      1009
      1010 ---
      1011
      1012 # ═══════════════════════════════════════
      1013 # ACT 10 — ABLATION STUDY & RESULTS FRAMEWORK
      1014 # ═══════════════════════════════════════
      1015
      1016 ---
      1017
      1018 ## SLIDE 34 — WHAT WE ARE MEASURING AND WHY
      1019
      1020 **VISUAL:**
      1021 Two metric cards, large, centred:
      1022
      1023 Card 1 — F1 SCORE:
      1024 "The harmonic mean of precision and recall.
      1025 Penalises both missing real planets (low recall) AND wrongly flagging false positives (low precision).
      1026 Range: 0 to 1. Higher is better."
      1027
      1028 Card 2 — AUC-ROC:
      1029 "Area Under the Receiver Operating Characteristic Curve.
      1030 Measures how well the model ranks true planets above false alarms at all possible thresholds.
      1031 Range: 0.5 (random) to 1.0 (perfect). Higher is better."
      1032
      1033 Below both cards: "We do NOT use raw accuracy. Under 41/59 class split, a model predicting everything FALSE POSITIVE scores 59% accuracy — and catches zero planets."
      1034
      1035 **ON-SCREEN TEXT:**
      1036 > Metrics chosen for class-imbalanced scientific classification:
      1037 > F1 Score — balances not missing real planets AND not flooding researchers with false alarms
      1038 > AUC-ROC — measures ranking quality across all thresholds
      1039 > Why not accuracy: trivially gamed by always predicting the majority class
      1040
      1041 **NARRATION:**
      1042 "The choice of metric is a methodological decision, not a formality. Under our 41/59 class split, raw accuracy is misleading. F1 score and AUC-ROC are the standard metrics in the exoplanet classification literature — AstroNet, Choudhary et al., and all c
           omparable papers report these. Our results table will use the same metrics, making our comparison directly meaningful."
      1043
      1044 **ANIMATION CUE:**
      1045 F1 card slides in from left. AUC-ROC card from right. Both animate open like playing cards. Accuracy card appears at bottom with a red X through it.
      1046
      1047 ---
      1048
      1049 ## SLIDE 35 — THE ABLATION STUDY
      1050
      1051 **VISUAL:**
      1052 A results table (to be filled with actual numbers after experiments run):
      1053
      1054 | Configuration | F1 | AUC-ROC | Notes |
      1055 |---|---|---|---|
      1056 | Zero-shot ViT | TBD | TBD | No adaptation |
      1057 | One-shot ViT | TBD | TBD | 1 example/class |
      1058 | Few-shot ViT | TBD | TBD | Small labelled set |
      1059 | LoRA r=4 | TBD | TBD | ~150K params |
      1060 | LoRA r=8 | TBD | TBD | ~600K params |
      1061 | LoRA r=16 | TBD | TBD | ~1.2M params |
      1062 | LoRA r=8 + RAG | TBD | TBD | Full system |
      1063
      1064 Plus: LoRA r=8, k=3 vs k=5 vs k=10 retrieval sweep.
      1065 Plus: TESS zero-shot generalisation (Kepler model on TESS data).
      1066
      1067 **ON-SCREEN TEXT:**
      1068 > Ablation Study — systematically isolating the contribution of each component
      1069 > Which paradigm is most effective? (zero-shot → few-shot → LoRA)
      1070 > Which LoRA rank is optimal? (r=4 vs r=8 vs r=16)
      1071 > Does RAG improve trust without hurting classification? (it should — RAG is post-hoc)
      1072 > Does the Kepler-trained model generalise to TESS zero-shot?
      1073
      1074 **NARRATION:**
      1075 "The ablation study is the backbone of the experimental contribution. By testing each configuration in isolation, we can attribute performance gains to specific design choices. The jump from zero-shot to LoRA quantifies how much domain adaptation matters
           . The rank sweep identifies the right capacity. The LoRA + RAG row shows whether the full pipeline adds scientific value beyond the classifier alone — which it should, since RAG does not modify predictions."
      1076
      1077 **ANIMATION CUE:**
      1078 Table rows slide in one by one. TBD cells pulse with a placeholder animation. Arrows point to key comparison pairs (zero-shot vs LoRA, r=4 vs r=8 vs r=16).
      1079
      1080 ---
      1081
      1082 ---
      1083
      1084 # ═══════════════════════════════════════
      1085 # ACT 11 — IMPACT & SCIENTIFIC CONTRIBUTION
      1086 # ═══════════════════════════════════════
      1087
      1088 ---
      1089
      1090 ## SLIDE 36 — WHY THIS MATTERS TO SCIENCE
      1091
      1092 **VISUAL:**
      1093 A world map with NASA facility locations highlighted. Overlay: TESS satellite data stream counter showing "~20,000 new KOIs/year". Below: three impact statements as cards.
      1094
      1095 Card 1: "SCALABILITY — Process thousands of candidates in hours, not years"
      1096 Card 2: "TRUST — Grounded explanations let scientists verify AI decisions"
      1097 Card 3: "GENERALISATION — Kepler model tested on TESS zero-shot (same physics, different instrument)"
      1098
      1099 **ON-SCREEN TEXT:**
      1100 > TESS is generating ~20,000 new planet candidates per year.
      1101 > Manual expert vetting cannot scale.
      1102 > An explainable AI classifier that cites known confirmed systems:
      1103 > — Is transparent enough for peer review
      1104 > — Is fast enough for the survey scale
      1105 > — Is honest enough to flag its own uncertainty
      1106 > This is what responsible AI in science looks like.
      1107
      1108 **NARRATION:**
      1109 "The scientific impact of this work extends beyond the Kepler dataset. TESS — NASA's current planet-hunting telescope — is generating new candidates at a rate that will never be manually vetted fast enough. The system built here, with its explainability
           layer and honest uncertainty quantification, is a prototype for how AI can responsibly assist at this scale. Not by replacing astronomers, but by giving them a reliable, auditable first-pass that they can trust."
      1110
      1111 **ANIMATION CUE:**
      1112 World map zooms in on NASA centres. TESS counter ticks up. Three cards pop in with delay. Final text fades in: "Not replacing astronomers — giving them a trusted first pass."
      1113
      1114 ---
      1115
      1116 ## SLIDE 37 — NOVEL CONTRIBUTION SUMMARY
      1117
      1118 **VISUAL:**
      1119 A three-layer stack diagram:
      1120
      1121 Layer 1 (bottom, grey): "Prior work — CNN / ViT classifiers — labels only, no explanation"
      1122 Layer 2 (middle, blue): "This project — ViT + LoRA — parameter-efficient, systematic evaluation"
      1123 Layer 3 (top, gold): "This project — Agentic RAG layer — scientifically grounded explanations citing confirmed systems"
      1124
      1125 Arrow pointing upward: "Each layer adds something that did not exist before."
      1126
      1127 **ON-SCREEN TEXT:**
      1128 > Three contributions:
      1129 > 1. Systematic four-paradigm ViT evaluation (zero-shot → LoRA) on Kepler GAF images
      1130 > 2. LoRA rank sweep (r=4, 8, 16) — efficiency vs performance trade-off documented
      1131 > 3. First agentic RAG explanation layer for exoplanet classification
      1132 >    — retrieves real confirmed systems — not generated facts
      1133 >    — flags its own uncertainty
      1134 >    — No prior exoplanet classifier does this
      1135
      1136 **NARRATION:**
      1137 "The novel contribution has three layers. The systematic evaluation framework comparing four adaptation paradigms on this specific dataset. The LoRA rank analysis, which documents the efficiency-performance trade-off for Kepler transit classification. An
           d the agentic RAG explanation pipeline — which, to the best of our knowledge, is the first system to attach scientifically grounded, retrieval-backed explanations to exoplanet classification predictions. That last layer is what makes this a research cont
           ribution rather than an application of existing techniques."
      1138
      1139 **ANIMATION CUE:**
      1140 Layers build upward one by one. Each layer glows as it settles. Arrow pulses upward. Gold top layer has a brief shimmer effect.
      1141
      1142 ---
      1143
      1144 ## SLIDE 38 — CLOSING SLIDE
      1145
      1146 **VISUAL:**
      1147 Return to the starfield from Slide 01. This time, several stars are highlighted with green rings — confirmed planets. The tiny dot of a planet crosses one of them again. Fade to the project title and a final statement.
      1148
      1149 **ON-SCREEN TEXT:**
      1150 > "The universe holds perhaps 100 billion planets in this galaxy alone."
      1151 > "Finding them is a search problem."
      1152 > "Trusting the result is an explanation problem."
      1153 > "This project addresses both."
      1154 >
      1155 > Explainable Exoplanet Transit Classification
      1156 > Ahmed Fayyaz Butt | COM748 | Ulster University | 2026
      1157
      1158 **NARRATION:**
      1159 "There may be 100 billion planets in our galaxy. Kepler watched a tiny corner of it and found thousands. The real challenge ahead is not the discovery — TESS, Plato, and future missions will find hundreds of thousands more. The challenge is knowing which
            signals to trust, and being able to explain why. That is what this project is about."
      1160
      1161 **ANIMATION CUE:**
      1162 Starfield returns slowly. Green rings appear on several stars (animated, one by one). Planet crosses again. Text fades in line by line. Logo fades in last. Hold for 3 seconds then end.
      1163
      1164 ---
      1165
      1166 ---
      1167
      1168 # APPENDIX — CANVA PRODUCTION NOTES
      1169
      1170 ## Colour Palette
      1171 - Background: #0A0E27 (deep space navy)
      1172 - Confirmed planets: #22C55E (green)
      1173 - False positives: #EF4444 (red)
      1174 - Uncertain/flagged: #F59E0B (amber)
      1175 - LoRA / architecture: #3B82F6 (blue)
      1176 - RAG / retrieval: #F97316 (orange)
      1177 - Text: #F8FAFC (near white)
      1178 - Accent: #FBBF24 (gold — novel contributions)
      1179
      1180 ## Font Recommendations
      1181 - Headings: Space Grotesk Bold or Inter Bold
      1182 - Body text: Inter Regular
      1183 - Code/data: JetBrains Mono or Fira Code
      1184
      1185 ## Animation Style
      1186 - Use "Breathe" or "Fade" for text appearances
      1187 - Use "Slide" for diagram elements building in
      1188 - Use "Rise" for cards popping up
      1189 - Keep transitions under 0.5s — fast is professional
      1190
      1191 ## Export Settings
      1192 - Video: MP4, 1920×1080, 24fps
      1193 - Animated GIF: 1280×720 (file size limit)
      1194 - For professor presentation: use Canva's present mode directly (no export needed)
      1195
      1196 ## Alternative Tools
      1197 - Genially (genially.com) — free, interactive, clickable infographic — better for self-paced professor demo
      1198 - Google Slides — free, animations built in, shareable link
      1199 - Prezi (prezi.com) — free tier, zoom animations, good for storytelling
      1200
      1201 ---
      1202
      1203 *Script written for dissertation presentation: Explainable Exoplanet Transit Classification Using Vision Transformers, LoRA, and Retrieval-Augmented AI on NASA Kepler Light Curves. Ahmed Fayyaz Butt, COM748, Ulster University, 2026.*
