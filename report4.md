To target the **Main Technical Tracks of IEEE ICASSP 2027** (Speech and Language Processing [SLP], Machine Learning for Signal Processing [MLSP], Audio and Acoustic Signal Processing [AASP], and Applied Signal Processing Systems [ASPS]), your submission must stand on **pure academic signal processing and architectural rigor**.

In high-tier IEEE conferences, clinical or industrial utility is treated *merely as an optional motivating example*. What reviewers actually score is whether you have solved a fundamental mathematical, representation, or computational bottleneck using universally accepted benchmarks.

Four standard, heavily published ICASSP research topics directly mirror the core technical challenges you observed during your internship (code-switching breakdowns, streaming latency, prosodic flattening, and edge power profiles). For each, the focus is placed on a **formal research methodology, industry-standard benchmark datasets, and standard academic evaluation metrics.**

---

### TOPIC 1: Zero-Shot Code-Switched Speech Synthesis

**Target Track:** Speech and Language Processing (SLP) – *Text-to-Speech Synthesis & Voice Cloning* *How your project relates (Optional Motivation):* This generalizes your observation of Coqui XTTS breaking down, babbling, or losing attention when switching between English and Hindi/Hinglish.

#### 1. The Core Academic Research Problem

Autoregressive (AR) zero-shot TTS architectures suffer from severe **Cross-Attention Alignment Drift** and prosodic degradation when synthesizing intra-sentential code-switched text (e.g., alternating between Latin/ASCII and Brahmic/Devanagari scripts). Because the attention matrices are trained predominantly on monolingual datasets, script density mismatches cause attention collapse, resulting in acoustic hallucinations, trailing babble, or failure to emit clean End-of-Sequence (EOS) tokens.

#### 2. Research-Based Methodology (What to develop & test)

* **Phoneme-Aligned Cross-Attention Entropy Regularization:** Instead of feeding raw text or characters, map all multilingual text into a **Universal Phonetic Representation** (e.g., International Phonetic Alphabet [IPA] or X-SAMPA) using a shared embedding bottleneck.
* Introduce a custom **Attention Entropy Loss Function ($\mathcal{L}_{\text{entropy}}$)** during fine-tuning. This mathematically penalizes high-entropy (scattered) attention matrices at script-transition boundaries, forcing the neural decoder to maintain monotonic alignment across code-switching points without manual text padding.

#### 3. Standard Academic Datasets

* **SEAME (South East Asia Mandarin-English):** The global academic standard benchmark for spontaneous code-switched speech processing.
* **FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech):** Google’s 102-language benchmark; standard for evaluating zero-shot multilingual voice cloning.
* **CS-Hindi-English / IndicTTS:** Public, standardized Indic speech corpora maintained by IIT Madras and AI4Bharat.

#### 4. Standard Evaluation Metrics

* **MCD (Mel-Cepstral Distortion) $[\text{dB}]$:** Measures acoustic spectral fidelity against ground truth (lower is better).
* **SIM-R & SIM-O:** Cosine similarity of speaker embeddings (extracted via WavLM or ECAPA-TDNN) between the reference speaker audio and the generated speech to prove identity preservation.
* **ASR-WER / CER (Word/Character Error Rate):** Passing the synthesized audio through an external, frozen ASR (e.g., Whisper-Large) to objectively measure intelligibility and verify that code-switched words were not skipped or hallucinated.
* **CMOS (Comparative Mean Opinion Score):** Subjective blind evaluation by native bilingual speakers grading naturalness on a -3 to +3 scale.

---

### TOPIC 2: Low-Latency Streaming Speech-to-Speech Translation (S2ST)

**Target Track:** Machine Learning for Signal Processing (MLSP) / SLP – *Spoken Language Translation & Streaming Architectures* *How your project relates (Optional Motivation):* This generalizes your attempt to use SeamlessM4T v2 and your struggles with high Time-To-First-Audio (TTFA) due to structural tag blockades.

#### 1. The Core Academic Research Problem

Direct Speech-to-Speech Translation (S2ST) foundation models (such as Meta’s SeamlessM4T UnitY2 architecture) operate offline: they require the full source audio sentence before emitting target speech units. Transforming these into **Simultaneous / Streaming S2ST** models leads to a strict trade-off: firing speech tokens too early results in semantic hallucination due to missing context, whereas waiting for context inflates the **Ear-Voice Span (EVS)** beyond real-time conversational limits.

#### 2. Research-Based Methodology (What to develop & test)

* **Continuous Integrate-and-Fire (CIF) with Speculative Causal Decoding:** Implement a non-autoregressive acoustic unit streaming module. Place a mathematical **CIF boundary predictor** over the speech encoder that accumulates acoustic weights frame-by-frame.
* When the accumulated weight crosses a learned threshold $\tau$, it dynamically fires an acoustic unit to the vocoder while using a **causal cross-attention mask** over future frames. You train this using a joint optimization loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{translation}} + \lambda \cdot \mathcal{L}_{\text{latency}}$, proving mathematically that your model optimizes the EVS-to-BLEU Pareto frontier.

#### 3. Standard Academic Datasets

* **CVSS (Common Voice Speech-to-Speech Translation):** Created by Google; the industry-standard benchmark for direct speech-to-speech translation across 21 languages (includes English $\leftrightarrow$ Indic language pairs).
* **CoVoST 2:** A large-scale multilingual speech translation benchmark created by Meta.
* **Europarl-ST:** Standard benchmark for simultaneous speech translation.

#### 4. Standard Evaluation Metrics

* **BLASER 2.0:** Meta’s state-of-the-art, reference-free AI evaluation metric specifically designed to score the semantic and acoustic quality of direct speech-to-speech translation.
* **ASR-BLEU / sACREBLEU:** Transcribing the generated speech output via ASR and computing BLEU scores against ground-truth text translations.
* **EVS (Ear-Voice Span) $[\text{ms}]$:** The exact time delay between when a word is spoken in the source audio and when its translation is synthesized in the target audio.
* **AL (Average Lagging) & AP (Average Proportion):** Standard mathematical metrics for simultaneous speech translation latency.

---

### TOPIC 3: Disentangled Prosody and Affect Transfer in Speech Dialog Models

**Target Track:** Multimedia Signal Processing (MMSP) / SLP – *Speech Emotion Recognition & Prosodic Modeling* *How your project relates (Optional Motivation):* This generalizes your challenge of "robotic empathy" where the AI understands semantic distress in text but speaks in a flat, detached voice.

#### 1. The Core Academic Research Problem

Current audio-in/audio-out dialogue models treat discrete speech tokens (e.g., EnCodec or Mimi units) as a flattened, 1D acoustic sequence. This mathematically entangles semantic linguistic content with speaker identity and emotional prosody. When dialogue models generate responses, they suffer from **"Prosodic Flattening"**—regressing to an average, neutral training prosody that fails to mirror the user's emotional urgency or affective state.

#### 2. Research-Based Methodology (What to develop & test)

* **Information-Theoretic Disentanglement via Residual VQ-VAE:** Design a neural speech tokenizer that splits acoustic representations into three orthogonal subspaces using mutual information minimization: a **Semantic Space** ($Z_s$), a **Speaker Identity Space** ($Z_i$), and a **Prosodic-Affective Space** ($Z_p$, capturing $F_0$, energy, and jitter).
* Implement an **Acoustic Affective Biasing Layer**: During dialogue generation, the language model predicts only the semantic tokens ($Z_s$), while a parallel lightweight graph predictor calculates the target prosodic trajectory ($Z_p$) conditioned on the input speech emotion. The vocoder synthesizes them together, achieving true zero-shot emotional mirroring.

#### 3. Standard Academic Datasets

* **IEMOCAP (Interactive Emotional Dyadic Motion Capture):** The universal gold standard academic dataset for emotion recognition and affective speech processing.
* **ESD (Emotional Speech Dataset):** High-quality multilingual dataset (English/Mandarin) with 350,000 utterances across distinct emotional states (Happy, Sad, Angry, Surprise, Neutral).
* **MSP-Podcast:** Large-scale real-world emotional speech dataset widely used in recent IEEE SER papers.

#### 4. Standard Evaluation Metrics

* **PCC (Pearson Correlation Coefficient) & RMSE:** Computed on the fundamental frequency ($F_0$) and energy contours between the target emotion contour and the synthesized speech contour.
* **SER-UAR (Unweighted Average Recall) $[\%]$:** Passing the generated audio into a standardized, frozen Speech Emotion Recognition classifier (e.g., WavLM-SER) to verify if the output audio is mathematically classified as the target emotion (e.g., "Empathetic/Comforting" vs "Neutral").
* **MCD & FFE (F0 Frame Error):** Standard spectral and pitch distortion metrics.
* **MOS-P (Prosody Mean Opinion Score):** Subjective grading focused strictly on emotional expressiveness and vocal inflection.

---

### TOPIC 4: Hardware-Aware Quantization & Speculative Decoding for Edge Speech Foundation Models

**Target Track:** Applied Signal Processing Systems (ASPS) – *Efficient Edge Implementation & Hardware-Co-Design* *How your project relates (Optional Motivation):* This generalizes your experiments running Blackwell GPUs, CPU-first initialization workarounds, VRAM limits, and tracking Joules/turn.

#### 1. The Core Academic Research Problem

Deploying large Speech Foundation Models (such as Whisper-Large, SeamlessM4T, or multimodal audio-LLMs) on edge workstations introduces severe VRAM and memory-bandwidth bottlenecks. While standard **Post-Training Quantization (PTQ)** works well for text LLMs, speech transformer activations exhibit massive, continuous outlier spikes in their acoustic cross-attention layers. Applying standard INT4/INT8 quantization to speech models causes catastrophic degradation in Word Error Rate (WER) and spectral reconstruction.

#### 2. Research-Based Methodology (What to develop & test)

* **Outlier-Aware Mixed-Precision Quantization (W4A8-Speech):** Formulate a mathematically derived quantization scheme that applies FP8 or INT8 scaling dynamically only to acoustic attention matrices containing activation outliers (> $99.9\text{th}$ percentile), while aggressively compressing feed-forward linear weights to INT4.
* **Acoustic Speculative Decoding:** Implement a lightweight, 100M-parameter "Draft Acoustic Model" that speculatively predicts 4–8 audio codec tokens ahead of time, which are then verified in parallel by the large Speech Foundation Model. This fundamentally bypasses memory-bandwidth bottlenecks on edge GPUs without altering the final output math.

#### 3. Standard Academic Datasets

* **LibriSpeech (Clean & Other - Test Sets):** The absolute universal benchmark for evaluating speech model accuracy and computational speedups.
* **Common Voice (Mozilla):** Multilingual evaluation sets to prove quantization does not disproportionately degrade low-resource or Indic languages.
* **MLPerf Inference Benchmark Suite:** Standard framework for edge AI hardware latency/energy benchmarking.

#### 4. Standard Evaluation Metrics

* **RTF (Real-Time Factor):** The ratio of processing time to audio duration ($\text{RTF} = \frac{\text{Processing Time}}{\text{Audio Duration}}$). An $\text{RTF} < 1.0$ is mandatory for real-time edge streaming; aim to prove your methodology drops RTF from $0.8$ down to $0.2$.
* **Energy per Token / Turn $[\text{Joules}]$:** Measured rigorously using NVML, Intel RAPL, or external hardware power meters.
* **Memory Bandwidth Utilization $[\text{GB/s}]$ & Peak VRAM $[\text{MB}]$:** Measuring the exact hardware footprint reduction.
* **$\Delta$WER / $\Delta$CER $[\%]$:** The exact degradation in recognition or synthesis accuracy compared to the unquantized FP16 baseline model (proving that your quantization achieved massive speedups with $< 0.5\%$ accuracy loss).

---

### SUMMARY REFERENCE MATRIX FOR ICASSP 2027

| Topic # & Title | Target Track | Core Methodology (Research) | Standard Datasets | Standard Evaluation Metrics |
| --- | --- | --- | --- | --- |
| **1. Code-Switched Zero-Shot TTS** | **SLP** (Speech Synthesis) | Cross-Attention Entropy Regularization & Phoneme Bottlenecks | SEAME, FLEURS, IndicTTS | MCD, SIM-R/O, ASR-WER, CMOS |
| **2. Low-Latency Streaming S2ST** | **MLSP / SLP** (Speech Translation) | Continuous Integrate-and-Fire (CIF) & Speculative Causal Masking | CVSS, CoVoST 2, Europarl-ST | BLASER 2.0, ASR-BLEU, EVS (ms), AL |
| **3. Disentangled Prosody & Affect** | **MMSP / SLP** (Emotion/Prosody) | Residual VQ-VAE Information Bottleneck & Graph Prosody Biasing | IEMOCAP, ESD, MSP-Podcast | SER-UAR (%), PCC / RMSE ($F_0$), MOS-P |
| **4. Edge Quantization & Speculative Decoding** | **ASPS** (Hardware & Systems) | Outlier-Aware W4A8 Quantization & Acoustic Speculative Drafting | LibriSpeech, Common Voice, MLPerf | RTF, Joules/Turn, Peak VRAM, $\Delta$WER |

### How to Proceed with Your Team

Pick **one** of these four formal topics. If you select **Topic 1 or Topic 4**, you can directly utilize the Python codebase, baseline models (Whisper/XTTS/Gemma), and NVIDIA RTX PRO 5000 hardware you have already set up—simply swap out your internal clinical test prompts for the standardized datasets (like FLEURS, LibriSpeech, or CVSS) and run the standard mathematical evaluation metrics!
