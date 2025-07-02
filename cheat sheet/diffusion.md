

# Diffusion 


- Diffusion vs gan
    - More stable and can learn more general data distribution 
    - More expansive

## Basic Structure


**DDPM**

- Forward: Add noises step by step to the input image
- Backward: pridict noise **reversely by sampling**. It is a **stocastic estimation**
    - Depend on previous **sampling**** xt **
    - Gaussian noise sampling

- What is noise scheduler:  noise variance 𝛽𝑡 over 𝑇 steps
    - In inference, will add a  **random noise at output**
        - Each time outputs / generated figures are different 
        - Make the process stocastic ⇒ more accurate learn 


- score function: learned the **distribution of data**. It is learned to match the distribution of data 
![Image](./media/1742512715608_d_22405834-30de-804b-bb90-fc28c406b734.png)


- DDIM:
    - difference: forward is the same. Backward is **computed instead of sampling and estmated**
        - Only add noise at **certain steps **
        - Not adding any **random variables **during both training and inference so results are always the same 


- How to speed up
    - DDIM
    - Latent Diffusion: compress to a **lower dimension space**
    - Distillation: train a small set
    - More efficient noise scheduler like **cosine noise**

- What is conditioned diffusion, difference between classifer-free classifer-based
    - Add more information
    - classifer-based: **pretrained network ⇒ **gradient for diffusion 
    - classifer-free: **interpolote the result **w/t info

- SDXL vs flux 
    - flux: transformer encoder-based ⇒ More Global 
    - SDXL: partial transformer based ⇒ **attention based**

## T2I Generation


### UNet


Reference link: [https://medium.com/@onkarmishra/stable-diffusion-explained-1f101284484d](https://medium.com/@onkarmishra/stable-diffusion-explained-1f101284484d)

Structure

![Image](./media/AD_4nXdeKmDMuoRTUMd1ARI3bgOLMyrV6lnBx6JcT1Md5ZJA39mLT8GcEr_0Q0W-7jEZiRRDAeMEp2_W_sIbZDK5OI2Th4q6kXiWcjvUkz7AHet9zWPwnKkerhMRhMlYXwNyPQRNmbbHT_lhQ8YAsDJrVjPJb4OD_22405834-30de-80c9-b721-f23d4e270b42)


- VAE: into **latent space**, i.e: a smaller or lower dimension space ⇒ input of UNet; decoder: from latent space back to origin;
    - only training need to reflect into latent space. Inference can directly start from noise/hint and denoise(training needs to add noise)

- text embedding: Using CLIP text embedding ( when inference, CLIP will project text to an embedding)
- UNet: add conditioned text to the net ( like another encoder and decoder
    - train: input: encoded image input
    - infer: input: encoded noise / guided mask

### DiT


Link: [DiT](https://www.wpeebles.com/DiT)

![Image](./media/image_22405834-30de-80e7-b66e-e77dd98a9293.png)


### Flow Matching 


## Personalization Generation


### Dream Booth


![Image](./media/AD_4nXeOF5q0PqL55lJ6I43vjMXwGRMKTgvSzNdf3GYAOn_KRSq4Pdo6isT-btgMsvxg-4_tL-N781TCtKzJKyAuiAakKoSOv3qn-qHlzxTFhz9OIfENZD_bz_58GH04_PwKojR4nFfJFH22VRpguhEcaBI3Dnk_22405834-30de-80b4-af17-f67c8fa3e86c)


leverage between fine-tuned learned and original network generation ability

Solution: use two part of loss

- compared with original class prior image
- let the generation after fine-tuned compared with generation before fine-tuned
how to describe the class prior

? target: weaker prior

Use sequence of characters to replace specific descriptive word

⇒ sample from tokenizer

decode to sequence

### IP Adaptor


add image embedding into  **another cross attention module **

## Low-Level Guidance


### ControlNet


- A hard modulate over output: Like add a **modulator in each output feature map**
- Add layers of `1x1 conv` + `copy trained nueral layers` to learn hard mask 
- Trained parameters
    - zero-init conv: init as 0 to  **avoid adding noise at start **
    - copied layer: all layers in **encoder **
        - The encoder block will use **corresponding zeor init conv and added to correspoinding decoder **


- Advantage
    - Precise 
    - Support multiple conditioning  

- Disadvantage 
    - Expensive: make inference slower even after trained 
    - Memory intensive: one - one ⇒ have to store motiply resutls

![Image](./media/2bff698f99ef8960264e52492de83a6_22405834-30de-80da-a252-c9e6ca96a0ac.jpg)


![Image](./media/image_22405834-30de-802d-8952-d37b9490e387.png)

