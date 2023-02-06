# SCALE-UP

We release our codes and several models for demonstration. 
We store the poisoned datasets and poisoned models for BadNets and WaNet in   https://www.dropbox.com/sh/lhgr6g8v7lohao2/AAArQpt5Vty3O0C4rdIr9s-ua?dl=0  and https://www.dropbox.com/sh/99cmqkqfcqpg555/AAAhyOSmP2tjJsRx0u3ViSLwa?dl=0 

You can run: 
```bash 
python ./test.py 

```
to reimplement the results for WaNet

To reimplement other results, you should first download the BadNets, WaNet folder from above links in ./ dictatory. Then you can use  
```bash
python torch_model_wrapper.py 
```
to extract SPC scores for different poisoned models. The SPC scores will be stored in the saved_np/ file.

Then you can change the path in process("saved_np/WaNet/tiny_bd.npy") to test SCALE-UP for other attacks.  

We will upload other datasets and poisoned images lately, or you can craft poisoned samples and models using BackdoorBox 
