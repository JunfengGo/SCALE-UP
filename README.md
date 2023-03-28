This is the official implementation of our paper '[SCALE-UP: An Efficient Black-box Input-level Backdoor Detection via Analyzing Scaled Prediction Consistency](https://openreview.net/pdf?id=o0LFPcoFKnr)', accepted in ICLR 2023. This research project is developed based on Python 3 and Pytorch, created by [Junfeng Guo](https://personal.utdallas.edu/~jxg170016/) and [Yiming Li](http://liyiming.tech/).


## Reference
If our work or this repo is useful for your research, please cite our paper as follows:
```
@inproceedings{guo2023scale,
  title={SCALE-UP: An Efficient Black-box Input-level Backdoor Detection via Analyzing Scaled Prediction Consistency},
  author={Guo, Junfeng and Li, Yiming and Chen, Xun and Guo, Hanqing and Sun, Lichao and Liu, Cong},
  booktitle={ICLR},
  year={2023}
}
```

## Implementation
We release our codes and several models for demonstration. 
We store the poisoned datasets and poisoned models for BadNets and WaNet in [DropBox1](https://www.dropbox.com/sh/lhgr6g8v7lohao2/AAArQpt5Vty3O0C4rdIr9s-ua?dl=0) and [DropBox2](https://www.dropbox.com/sh/99cmqkqfcqpg555/AAAhyOSmP2tjJsRx0u3ViSLwa?dl=0). We also generate several SPC value for different attacks, which are saved in saved_np.

You can run: 
```bash 
python ./test.py 

```
to reimplement the results for WaNet.

To reimplement other results, you should first download the BadNets, WaNet folder from above links in ./ dictatory. Then you can use  
```bash
python torch_model_wrapper.py 
```
to extract SPC scores for different poisoned models. The SPC scores will be stored in the saved_np/ file.

Then you can change the path in process("saved_np/WaNet/tiny_bd.npy") to test SCALE-UP for other attacks (e.g., ISSBA, TUAP). You can craft poisoned samples and models using [BackdoorBox](https://github.com/THUYimingLi/BackdoorBox).
