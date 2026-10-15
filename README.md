# Fine-Tuning-LLM
This is to fine tune the LLM  specifically for Data Structure and Algorithms .


NOTES FOR MYSLELF ABOUT THE CONCEPTS UTILISED.
----------------------------------------------
1 ) LORA MODULES AND HOW THEY ARE ATTACHED.
- - - - - - - - - - - - - - - - - - - - - - - 
LOW RANKING ADAPTIVE MODULES.
basically the problem we face right now  without lora is that ,if we want to change the weights of the parameters then we have to 
do it on a large scale which involves a lot of matrix multiplication of millions of rows of weights.

doing this wastes a lot of compute and is very time consuming.
 How LORA resolves this issue is by freezing the entire parameter's weight matrix. 
 LORA adds small brand new adapater parameters which are trainable parrele to

 LORA adapters are attached parallelly existing weights of the parameters.
 
 The new weights are given as:
 h=W+(alpha/r)*delta(W)

 alpha/r done for scalability purpose.

 The delta(W) is the change in the weights of the parameters. 
 delta(w) is where we divide the entire weight into 2 relatively smaller matrices, instead of one whole big d*k matrix
 delta(W) is =B@A where the B represents the d*r matrix and the the A matrix represent the r*k matrix
 (r->rank of the weight matrix and the d*k size of  original matrix)
 
 alpha represent how strongly are we changing the weights of the parameters , and the r is representing the rank of the weight matrix

 
2) ATTENTION AND THE MLP(mutli layer perceptron):
   - - - - - - - - - - - - - - - - - - - - - - -
  ATTENTION PART:
   For the current model used that is mistral 7B (7 Billion parameters) .
   AI Nueral Network is divided into multiple layers (here 32 layers from 0-31)
   Each layer has its own ATTENTION and the MLP.
   Every layer contains multiple(32 in this case) Attention heads.
   Every Attention head contains their own slice or portion  of the k,q,v matrix(key_proj,query_proj,value_proj)
   key_proj and the query_proj are used to retrieve the data.
   value_proj is used to hold the retrieved data.
   
   the output_proj(o_proj)  processes the values which are  obtained from different value_proj.


   LORA are attached to these target modules(ie the q_proj,k_proj,v_proj,o_proj )
   attaching to o_proj ,v_proj helps to change what information to extract and how it blends the informatin together.
   attaching to q_proj and the k_proj changes where the model looks and what information it extracts.

   MLP PART:
     the output of the  o_proj is passed  to the multi layer perceptron part of the respective nueral layer.
     the mlp consists of three matrices the up_proj,gate_proj,down_proj.
    up_proj : expands the given data into a large space 
    gate_proj: in the expanded space only turns on the information (data) which are relavent to the given context
    down_proj: decompresses it and sends to the next layer of nueral network.

    every nueral layer:
    attention(k,q,v,o)---->mlp(up_proj,gate_proj,down_proj)------->passed to next layer
    up_proj and the gate_proj work in parallel and the result is fed in the down_proj.
    
   
     

   

