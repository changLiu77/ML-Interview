

# Foundation Model


## Intro: application 


- object detection
- scenario silmulation
- Traditional solution
    - Component
        - perception
        - prediction: RL
        - planning

    - Train on: hand-craft data 

## Foundation Model Solution


 **large scale dataset** + LLM + cv ⇒ One for all

### LLM


- Used in 
    - Reason
    - Pridict
    - Interface: understand human instruction
    - simulation: generate more **text info for generate new scenarios**

- Techniques
    - prompt engineer
    - finetune: on small-dataset for specific tasks or in context: few-shot
    - Reinforce learning: RLHF: use human feedback to 

- Weakness
    - speed: model distillation / compression
    - precision
    - The training scearios are **too simple **⇒ generate more **complex** scenarios

### Vision


**User pretrained model / foundation model **

- Typical application: like a **feature extractor**
    - Large Vision model ⇒ embedding 
    - Learn embedding for other modalities: like point cloud ⇒ **space alignment / domain adaption**
    - Large object detaction ⇒ large segemantation ⇒ perception 

- Weakness 
    - Not zero-shot ⇒ error
    - Can not deal with other modality

**User Diffusion Model **

- Simulation: video generation ⇒ Merge other modalities as **cross attention input** ⇒ Supervised by other modalities 
### Multi-modal 


嵌套的model ⇒ May have several **intermediate modality**

Make use of the modal’s **reasoning **

- image ⇒ text: 
    - VQA: answer questions / understand image 
        - Train to give **action and planning **
        - Train to align **radar and cloud** with vision

    - Description: abundant info for next model 

# On-device ML


### Model Deployment 


**Basic Concept **

- Tensor
- Precision 
    - single precision: fp32 4bits
    - half precision: fp16 2bits 
    - int8: intenger 8 1bit (-128 - 127)

- GPU usage 
    - GPU video memory usage: the memory space of GPU, used for data and model
        - All parameters * 4 (fp32)
            - Network, optimizer 
            - Gradient(How to compute?)
                - In back propagate, what is stored, what is back?



    - GPU usage: computing units usage (like cuda): how many units’ **kernel is running**

- Computing times
    - FLOPs: 
        - Conv times `Cin * Cout * K * K * steps` ⇒ if filled to `(H + K , W + K)` (K = kernel size), ⇒ `Cin * Cout * K * K * H * W`
        - Conv add: one pixel `C_in * (K * K - 1) + Cin - 1 + 1 = C * K *  K` ⇒ All pixel `C_in * K*K * H* W` ⇒ All channel: `Cin*cout *K*K * H * W`

    - FLOPS: floating point operations per second ⇒ hardware performance
    - MAC

### TensorRT 


主要用于不同深度学习框架之间的模型转换和互操作性

不涉及模型的优化过程

为了提高模型在实际应用中的性能，解决其在推理时面临的高计算复杂度、大内存需求、并行性限制等问题，常用一些办法总结如下：
模型压缩技术：通过知识蒸馏、权重剪枝、量化等方法，减小模型体积和计算需求。这些技术通过精简模型结构或参数，减少不必要的计算负担，从而加快推理速度，降低存储和运行时内存需求。知识蒸馏特别值得注意，它通过将大模型的"知识"转移到小模型上，既保持了模型性能，又实现了模型大小的显著减少。
高效变体的开发：针对自注意力机制的高计算复杂度，开发了如Transformers的变体，通过修改自注意力机制来降低计算复杂度和内存需求。例如，使用稀疏注意力模式、局部注意力或低秩近似等技术，有效地减少了处理长序列时的计算负担。
硬件加速和专用推理引擎：利用TPU、GPU等专门设计的硬件加速器和优化的推理引擎（如TensorRT、ONNX Runtime），针对特定硬件平台的特性进行底层优化。这种优化可以显著提高推理速度，同时降低功耗，特别适合需要实时处理的应用场景。
动态量化和混合精度推理：通过在模型的不同部分使用不同的数据精度（如FP32、FP16、INT8）来平衡推理速度和模型精度之间的关系。动态量化特别适用于在运行时根据需求调整精度，而混合精度推理则可以在保持模型性能的同时加速模型的推理过程。

加速transformer进行推理

### Other tips


**cache**

- L2 cache: how many data stay in L2
    - More data in L2 better ⇒ less data fetch 
    - In block-data ⇒ large block can not goto L2 

- KV cache: restored computed **attention keys, values **
    - cached previous **token / embeddings**
    - K ⇒ **selectors**, V ⇒ **targeted value**, Q ⇒ **input **

- Application
    - For Large model, Like LLM, RAG
        - The out source info /  stored text is **precomputed as K and V **
        - The retrieved item 

    - For transformer based, but don’t have a **public attention block **⇒ **precomputed embedding **cache but not kv cache
        - precomputed pair: text-image for CLiP
        - When input, the **corresponding compared embedding **is fetched

