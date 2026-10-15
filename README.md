# Fine-Tuning-LLM
This repository contains scripts to fine-tune an LLM specifically for Data Structures and Algorithms (DSA).

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
h = Wx + (alpha / r) * (ΔW)x

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
