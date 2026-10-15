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
 How LORA resolves this issue is by freezing the entire weight matrix. Only few paramters are unfrozen and are changed

 LORA adapters are attached parallelly existing weights of the parameters.
 
 The new weights are given as:
 h=W+(alpha/r)*delta(W)

 The delta(W) is the change in the weights of the parameters. 
 delta(w) is where we divide the entire weight into 2 relatively smaller matrices, instead of one whole big d*k matrix
 delta(W) is =B@A where the B represents the d*r matrix and the the A matrix represent the r*k matrix
 (r->rank of the weight matrix and the d,k are the rows and columns of the original matrix)
 
 alpha represent how strongly are we changing the weights of the parameters , and the r is representing the rank of the weight matrix

 
2) ATTENTION AND THE MLP(mutli layer perceptron):
   - - - - - - - - - - - - - - - - - - - - - - -
   
   For the current model used that is mistral 7B (7 Billion parameters) .
   AI Nueral Network is divided into multiple layers (here 32 layers from 0-31)
   Each layer has its own ATTENTION and the MLP.
   Every layer contains multiple(32 in this case) Attention heads.
   Every Attention head contains thier own multiple  k,q,v (key_proj,query_proj,value_proj)
   key_proj and the query_proj are used to retrieve the data.
   value_proj is used to hold the retrieved data.
   
   the output_proj(o_proj)  processes the values which are  obtained from different value_proj.
   

