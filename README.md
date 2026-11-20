# Co²FuLL-Pro

> **Zheng Pan, Chengzhang Wang, Chaopeng Dong, Jingdong Guo, Yuxin Chen, Shouguo Yang, Yang Xiao, Limin Sun, Qiuyun Lv, Butian Huang.**
> *Co²FuLL-Pro: Resource-Aware Binary Code Similarity Detection with Evidence-Guided LLM Verification.*

**Co²FuLL-Pro** is the journal extension of our ASE 2025 framework [**Co²FuLL**](https://github.com/GentleCP/Co2FuLL-public). While Co²FuLL demonstrated that combining contextual retrieval with LLM verification substantially improves binary code similarity detection (BCSD), achieving strong accuracy in practice often required large commercial models (such as DeepSeek-V3 or GPT-4o).

Co²FuLL-Pro is designed for **local, resource-aware deployment** with a small-scale model (`Qwen2.5-7B-Instruct`, without fine-tuning or external code transmission), reaching **85.0% precision and 91.6% recall** on the XM benchmark and scaling to real-world IoT firmware vulnerability search.

*This repository releases the source code of Co²FuLL-Pro. The extended benchmark datasets and experimental artifacts will be made publicly available upon acceptance of the journal article.*

---

## ✨ What's New Since ASE 2025

Compared with the conference version ([Co2FuLL-public](https://github.com/GentleCP/Co2FuLL-public)), this repository introduces the post-retrieval verification architecture and tooling developed for the journal extension:

* **🎯 Calibrated Similarity Screening (CSS):** Replaces raw cosine cutoffs with Cross-Domain Similarity Local Scaling (CSLS) and call-graph neighborhood evidence, resolving high-confidence matches on CPU before invoking the LLM.
* **🔍 Evidence-Guided Post-Verification:** Instead of relying on a single-pass LLM decision, initial judgments from Pre-Verification are selectively reassessed by analyzing compilation effects and incorporating auxiliary binary evidence (alternative Ghidra/GSAT views, syntax/control-flow trees, and ELF metadata).
* **🧭 Modular & Inspectable Workflow:** Verification runs in clean stages (`route` → `fewshot` → `postverify`), and `co2full-inspect` lets you view the decompiled code, prompts, and model reasoning for any candidate pair.

```text
Co²FuLL Top-K CSV + Stripped ELFs
       │
       ▼  co2full-data / co2full-extract / co2full-features
  Prepared Assets (Hex-Rays, Ghidra/GSAT, HermesSim + CSLS, Call-Graph)
       │
       ├─► [Stage 1: CSS Screening]      co2full-route      ──► Direct decision (CPU)
       │
       ├─► [Stage 2: Pre-Verification]   co2full-fewshot    ──► Initial 7B Few-Shot Judgment
       │
       └─► [Stage 3: Post-Verification]  co2full-postverify ──► Evidence-Guided Reassessment
                                                                      │
                                                                      ▼
                                                       work/<run>/predictions.jsonl
```

---

## 🗺️ Repository Layout

* **`src/co2full/`** — CLI entry points, feature extraction pipeline, vLLM service manager, and evaluation routines.
* **`src/co2full/_engine/`** — CSS routing rules, prompt templates, compilation-effect checks, and post-verification stages.
* **`src/co2full/exporters/` & `tools/java/`** — IDA-Pro (Hex-Rays) scripts and Ghidra/[GSAT](https://github.com/sgfvamll/gsat) exporters for extracting pseudocode, CFGs, dependency graphs, and syntax trees.
* **`configs/` & `requirements/`** — Configuration template (`configs/xm.json`) and dependency files for the main environment, IDAPython, and HermesSim.
* **`scripts/`** — Utilities for preparing top-$K$ candidate inputs (`prepare_inputs.py`), building Java helpers (`prepare_tools.py`), and scoring firmware search cases (`evaluate_cases.py`).
* **`tests/`** — Unit tests for the verification pipeline and evaluation metrics.

For candidate retrieval and the public BinKit XM dataset, please refer to [Co2FuLL-public](https://github.com/GentleCP/Co2FuLL-public) and our [figshare archive](https://doi.org/10.6084/m9.figshare.30426451).

---

## 🛠️ Prerequisites & Configuration

### 1. Host & Inference Environment

Our experiments were conducted on Linux with **Python 3.10**, **CUDA 12.1**, **PyTorch 2.3.0**, and **vLLM 0.5.1**. Install the package from the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install .
```

For local LLM verification, install the inference dependencies and download [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct):

```bash
pip install -r requirements/inference-pins.txt
```

### 2. Binary Feature Extraction Toolchain *(Optional if using prepared assets)*

To extract features from raw ELF binaries, please also prepare:
* **IDA-Pro v9.3** (with Hex-Rays) and `requirements/ida-pins.txt`.
* **JDK 17** and **[GSAT v1.0](https://github.com/sgfvamll/gsat)**.
* **[HermesSim](https://zenodo.org/records/10369788)** with `requirements/hermes-pins.txt`.

### 3. Machine Configuration

Copy the template configuration and update the paths for your environment:

```bash
cp configs/xm.json configs/machine.json
```

```json
{
  "assets": "../work/xm",
  "model": "/path/to/Qwen2.5-7B-Instruct",
  "python": "../.venv/bin/python",
  "ida": "/path/to/ida/idat",
  "java_backend": "/path/to/jdk17/bin/java",
  "hermes_python": "/path/to/hermes-env/bin/python",
  "hermes_root": "/path/to/HermesSim",
  "hermes_checkpoint": "/path/to/checkpoint_19.pt"
}
```

---

## 🚀 Running the Verification Pipeline

### 📦 Step 1: Prepare Assets

Start with a top-$K$ candidate CSV from Co²FuLL retrieval and the corresponding stripped binaries. If you already have a prepared asset directory (`work/xm`), set `"assets"` in `configs/machine.json` and skip to **Step 2**.

```bash
export INPUTS=/path/to/your/inputs

# 1. Convert top-K CSV and stripped binaries into input manifests
python scripts/prepare_inputs.py \
    --candidates "$INPUTS/xm-full_top5-250515.csv" \
    --binaries "$INPUTS/Binkit-1.0-normal-strip-binaries" \
    --out work/inputs \
    --write-labels

# 2. Build the Java view exporter
python scripts/prepare_tools.py \
    --gsat "$INPUTS/gsat-1.0.jar" \
    --font "$INPUTS/NotoSans-Regular.ttf" \
    --out work/tools

# 3. Extract binary representations and finalize assets
co2full-data init \
    --pairs work/inputs/candidates.jsonl \
    --binaries work/inputs/binaries.jsonl \
    --examples "$INPUTS/few-shot-examples.json" \
    --tools work/tools \
    --out work/extraction

for step in idb ida pcode ghidra; do
    co2full-extract "$step" --config configs/machine.json --workspace work/extraction || exit 1
done

for step in hermes csls graph views; do
    co2full-features "$step" --config configs/machine.json --workspace work/extraction || exit 1
done

co2full-data finalize --workspace work/extraction --out work/xm
```

---

### ⚡ Step 2: Run CSS & Two-Stage LLM Verification

Screening (`co2full-route`) runs on CPU; Pre- and Post-Verification use the local vLLM service managed by `co2full-services`.

```bash
# Stage 1: Calibrated Similarity Screening
co2full-route --config configs/machine.json --run work/full

# Stages 2 & 3: Pre-Verification and Post-Verification
co2full-services start --config configs/machine.json
co2full-fewshot --run work/full
co2full-postverify --run work/full
co2full-services stop --config configs/machine.json
```

Final predictions are saved to `work/full/predictions.jsonl`.

---

### 📈 Step 3: Evaluation & Inspection

* **Evaluate retrieved candidates:**

  ```bash
  # Full Co²FuLL-Pro pipeline
  co2full-evaluate --run work/full --labels work/inputs/evaluation/labels.jsonl

  # Evaluate CSS + Pre-Verification only (w/o Post-Verification)
  co2full-evaluate --run work/full --labels work/inputs/evaluation/labels.jsonl --without-pv
  ```

* **Inspect candidate traces:**

  ```bash
  co2full-inspect --run work/full
  co2full-inspect --run work/full --pair-id <pair_id> --out work/pair-trace
  ```

* **Evaluate real-world firmware search cases:**

  ```bash
  python scripts/evaluate_cases.py /path/to/cases.jsonl
  ```

---

## 📚 Related Publication & Citation

If you find Co²FuLL-Pro or our conference work helpful in your research, please cite:

```bibtex
@inproceedings{dong2025co2full,
  title={Advancing Binary Code Similarity Detection via Context-Content Fusion and LLM Verification},
  author={Dong, Chaopeng and Guo, Jingdong and Yang, Shouguo and Li, Yi and Fang, Dongliang and Xiao, Yang and Chen, Yongle and Sun, Limin},
  booktitle={Proceedings of the IEEE/ACM International Conference on Automated Software Engineering (ASE)},
  pages={304--316},
  year={2025}
}
```

---

## ✉️ Contact

For questions about the code or reproduction details, please reach out to:

* **Zheng Pan**: `panzheng@hdu.edu.cn`
* **Chaopeng Dong**: `dongchaopeng@hdu.edu.cn`
