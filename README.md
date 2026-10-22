# Fine-Tuning-LLM
This repository contains scripts to fine-tune an LLM specifically for Data Structures and Algorithms (DSA).
# DSA Tutor LLM: Fine-Tuning Mistral-7B with QLoRA

Fine-tuning pipeline that adapts **Mistral-7B-Instruct-v0.3** into a Data Structures & Algorithms (DSA) tutor using 4-bit QLoRA, trained end-to-end on a single free Kaggle T4 GPU.

- **Adapter (v2):** [`GAnirudh711/mistral-7b-dsa-expert-v2`](https://huggingface.co/GAnirudh711/mistral-7b-dsa-expert-v2)
- **Datasets:** [`DSA_FINETUNING_LLM`](https://huggingface.co/datasets/GAnirudh711/DSA_FINETUNING_LLM) (20,000 examples) · [`DSA_FINETUNING_LLM_V2`](https://huggingface.co/datasets/GAnirudh711/DSA_FINETUNING_LLM_V2) (5,000 examples)

---

## Results

Evaluated on a 20-problem DSA benchmark (pass@1). Each model writes a Python function from a plain-English prompt; the code is extracted and run against unit tests.

| Model | Problems solved | pass@1 |
|---|---|---|
| Mistral-7B-Instruct-v0.3 (base) | 12 / 20 | 60% |
| **Fine-tuned (v2)** | **15 / 20** | **75%** |

**Method:** identical prompts and system prompt for both models, ChatML format, greedy decoding (reproducible), 3 to 5 assert-based tests per problem, 10-second timeout. Problems cover arrays, strings, hashing, stacks, intervals, DP, graphs (Dijkstra, cycle detection), binary search and two pointers.

**Read this with care:** 20 problems is a small sample, and some classic problems (Two Sum, Coin Change, etc.) overlap with the style of the LeetCode training data, which favours the fine-tuned model. Treat this as evidence of improvement, not proof. Per-problem outputs are in [`eval_results.json`](eval_results.json).

---

## Training Summary

| Item | Value |
|---|---|
| Base model | `Mistral-7B-Instruct-v0.3` (4-bit, via Unsloth) |
| Method | QLoRA, supervised fine-tuning (SFT) with TRL |
| LoRA config | r = 16, alpha = 32, dropout = 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Trainable parameters | 41,943,040 of 7.29B (0.58%) |
| Prompt format | ChatML (system / user / assistant) |
| Hardware | Kaggle, single Tesla T4 |

### Round 1: baseline
- 20,000 examples (10,000 Codeforces CoT + 10,000 Nvidia OpenCodeInstruct), 4,096-token context.
- 500 steps, effective batch size 8, learning rate 2e-4. This covered only about 4,000 examples, roughly 20% of one epoch.

### Round 2: continued training from the round-1 adapter
- 5,000-example refined dataset: 1,500 Nvidia + 500 Codeforces + 3,000 LeetCode editorial explanations (top-upvoted, code included). After a 2,048-token length filter (10.3% removed), **4,485 examples** remained.
- **561 steps (1 full epoch)**, learning rate 1e-4 with cosine schedule, 8-bit AdamW, 2 h 51 min.
- **Completion-only loss:** the question tokens are masked (`-100`) so the model learns only from the answer.
- Training loss: **0.81 → 0.63**.

---

## Pipeline

1. **Build the dataset:** stream sources from Hugging Face, convert to ChatML, filter by length, push to the Hub.
2. **Train:** load the 4-bit base model (or a previous adapter), attach or continue LoRA adapters, train with SFT.
3. **Evaluate:** compare the base model against the fine-tuned adapter using the same prompts and unit tests.
4. **Serve:** load the adapter from the Hub and chat through a Gradio interface (8,192-token context).

---

## Repository Structure

| File | Purpose |
|---|---|
| `llm-fine-tuning (2).ipynb` | Dataset construction and upload to the Hugging Face Hub |
| `fine-tuning-llm-2 (2).ipynb` | Round-1 training (QLoRA, Unsloth) |
| `fine-tuning-llm-v2.ipynb` | Round-2 training: continued from the round-1 adapter, completion-only loss |
| `fine-tuning-llm-3 (1).ipynb` | Inference, base-vs-adapter comparison and Gradio app |
| `eval_results.json` | Per-problem results of the 20-problem benchmark |

---

## Quick Start

Requires a GPU (a free Kaggle or Colab T4 works).

```python
!pip install unsloth

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="GAnirudh711/mistral-7b-dsa-expert-v2",
    max_seq_length=8192,
    dtype=None,
    load_in_4bit=True,
)
tokenizer = get_chat_template(tokenizer, chat_template="chatml")
FastLanguageModel.for_inference(model)

# Use the same system prompt as in training.
SYSTEM_PROMPT = (
    "You are an expert DSA (Data Structures & Algorithms) tutor and competitive "
    "programmer. Explain your approach and provide clean code."
)
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Explain the approach for the LeetCode problem "
                                "'Maximum Subarray', then give time and space complexity and code."},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda")
stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]

out = model.generate(**inputs, max_new_tokens=900, do_sample=False, eos_token_id=stop_ids)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

---

## What I Learned

The first round gave weak results. Diagnosing it led to these fixes in round 2:

| Problem found | Fix |
|---|---|
| Only about 20% of one epoch was seen | Train a full epoch on a smaller, cleaner set |
| Loss was computed on the whole sequence, including the prompt | Mask prompt tokens, train on the answer only |
| Training data was mostly code-only; no tutor-style explanations | Add LeetCode editorial explanations |
| Long examples were truncated, so the model never saw the end of its answer | Filter to a 2,048-token limit |
| Test prompts used different system prompts from training | Use one identical system prompt everywhere |

**Engineering note:** Unsloth's `train_on_responses_only` could not find the ChatML response marker with Mistral's tokenizer (there is no single `<|im_start|>` token), so every label was masked. I wrote custom masking that tokenizes the prompt part separately and masks it to `-100`.

---

## Known Limitations

- **Execution traces can be wrong.** The model writes correct code for binary search but can produce an incorrect step-by-step walkthrough, because a 7B model in 4-bit does not reliably simulate execution. Always verify traces against the code.
- **Code-only answers on casual prompts.** Because part of the training data is code-only, short prompts like "explain binary search" sometimes return only code. Prompts phrased as "Explain the approach for ..., then give complexity and code" work best.
- **Small benchmark.** See the note under Results.

## Future Work

- Larger benchmark with held-out problems that do not overlap with the training data.
- Training data with varied explanation-style prompts, to remove the code-only default.
- Trying a stronger code-specialized base model for better reasoning.

---

<details>
<summary><b>Concept notes: LoRA, attention and the MLP</b></summary>

### 1. LoRA modules and how they are attached

Without LoRA, a full fine-tune updates every parameter in the network, which is slow and expensive. LoRA keeps the base weight matrix **100% frozen** and adds small trainable adapter paths in parallel. Only these adapter parameters are trained.

The adapted output is:

`h = W·x + (alpha / r) · B·A·x`

- `r` is the rank (bottleneck dimension) of the adapter matrices.
- `alpha` is the strength ("volume") of the adapter's influence.
- `alpha / r` is a constant scaling factor. Keeping this ratio when you change `r` means the learning rate does not need to be re-tuned.
- The learned update `ΔW` is split into two low-rank matrices: `B` (d × r) and `A` (r × k), so `ΔW = B @ A`. Because `r` is small (e.g. 16), VRAM usage drops sharply.

### 2. Attention and the MLP

Mistral-7B has 32 sequential layers (0 to 31). Each layer has an attention block and an MLP block.

**Attention**
- `q_proj` and `k_proj` match tokens to each other (for example an opening bracket to its closing bracket, or a variable to its function). Mistral-7B has 32 query heads and uses grouped-query attention, so keys and values share 8 heads.
- `v_proj` gathers the content (the semantic meaning of tokens).
- `o_proj` combines the outputs of all heads into one context summary.
- LoRA on `q_proj` / `k_proj` changes *how* the model focuses and links code structure. LoRA on `v_proj` / `o_proj` changes *what* information is retrieved and how it is blended.

**MLP (SwiGLU, three matrices)**
- `up_proj` expands the 4,096-dimensional input into a 14,336-dimensional space, giving room to combine many DSA rules at once.
- `gate_proj` works in parallel and applies SiLU, switching relevant features on (syntax, algorithms) and irrelevant ones off.
- `down_proj` compresses the filtered result back to 4,096 dimensions for the next layer.

**Per-layer flow:** Input → Attention (q, k, v → o) → MLP (up & gate → down) → next layer

</details>

---

## Author

**G Anirudh**, B.Tech CSE (Data Science), VIT Vellore · [GitHub](https://github.com/GAnirudh2006)
NOTES FOR MYSELF ABOUT THE CONCEPTS UTILISED
----------------------------------------------
1) LoRA MODULES AND HOW THEY ARE ATTACHED
- - - - - - - - - - - - - - - - - - - - - - - 
LOW-RANK ADAPTATION (LoRA)
Without LoRA, performing a full fine-tune requires updating all parameters in the network. 
This involves massive matrix multiplication across billions of parameters, which is 
highly compute-intensive, slow, and expensive.

LoRA resolves this by keeping the entire base weight matrix 100% FROZEN. The base parameters 
never change. Instead, LoRA injects tiny, highly efficient adapter paths parallel to the 
existing weights. Only these new adapter parameters are trainable.

The mathematical adaptation is calculated as:
W new = Wx + (alpha / r) * (ΔW)x

- (alpha / r) is a constant scaling factor used for mathematical stability. If you change 
  the rank 'r' later to learn more complex patterns, maintaining this ratio ensures you 
  don't have to completely re-tune your global learning rate.
- alpha represents the "volume" or strength of the adapter's influence over the base model.
- r represents the 'Rank' (bottleneck dimension) of the adapter matrices.
- ΔW represents the learned weight update. To save memory, a giant d x k matrix update 
  is split into two low-rank matrices: B (size d x r) and A (size r x k). 
  Therefore, ΔW = B @ A. Because the inner dimension 'r' is very small (e.g., 16), it drastically 
  reduces VRAM usage.

2) ATTENTION AND THE MLP (MULTI-LAYER PERCEPTRON)
- - - - - - - - - - - - - - - - - - - - - - -
ATTENTION PART:
- The base model used here (Mistral 7B) features a neural network split into 32 sequential 
  layers (numbered 0 to 31). Each layer contains its own Attention Block and MLP Block.
- Within each layer's Attention Block, the workload is distributed across 32 parallel 
  Attention Heads.
- Each Attention Head operates on its own dedicated slice of the layer's overall q_proj (Query), 
  k_proj (Key), and v_proj (Value) matrices.
  * q_proj & k_proj: Act as index matchers. They figure out relationships between tokens 
    (e.g., matching an open bracket to a closed bracket, or a variable name to its function).
  * v_proj: Acts as the content gatherer, extracting the raw semantic meaning of the tokens.
  * o_proj (Output Projection): The final mixer. It takes the fragmented outputs from all 
    32 attention heads, glues them back together, and blends them into a unified context summary.

LoRA Attachment Context:
- Attaching LoRA to q_proj and k_proj adjusts HOW the model focuses and links code structures.
- Attaching LoRA to v_proj and o_proj adjusts WHAT factual code information is retrieved and 
  how it is blended down the line.

MLP PART:
The unified summary from o_proj is handed directly to the MLP (Thinking) department of the layer. 
The MLP acts as the reasoning engine and uses a 3-matrix layout (SwiGLU):
- up_proj: Takes the 4,096-dimensional input and explodes it into a massive 14,336-dimensional 
  vector space. This acts like a giant whiteboard, giving the model massive mathematical room 
  to cross-reference multiple DSA logic rules simultaneously without clashing.
- gate_proj: Operates in parallel with up_proj. It acts as a router/filter, applying a SiLU 
  activation function to turn ON relevant slots (like syntax/algorithms) and turn OFF 
  irrelevant slots (like conversational filler).
- down_proj: Takes the massive, filtered 14,336-dimensional whiteboard results and compresses 
  them back down into a dense 4,096-dimensional summary package so it can fit into the next layer.

Data Flow Pipeline Per Layer:
Input -> Attention (q, k, v -> o) -> MLP (up & gate -> down) -> Sent to Next Layer
