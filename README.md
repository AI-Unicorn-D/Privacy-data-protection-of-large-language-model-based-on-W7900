# Privacy-data-protection-of-large-language-model-based-on-W7900
Hardware: AMD Radeon PRO W7900 GPU for model training and inference High-performance servers Storage systems Networking infrastructure 

Software: AI frameworks (e.g., TensorFlow, PyTorch) Speech recognition and text-to-speech software Programming languages (Python, C++) Operating systems (Ubuntu 22.04) Docker for containerization and easy deployment
    
Build the environment
*When building wsl, pay attention to the compatibility between different versions of the software
Compatibility matrices (WSL) — Use ROCm on Radeon GPUs (amd.com)

1、Ensure that WSL is installed before proceeding with ROCm installation.
Open PowerShell or Windows Command Prompt in administrator mode by right-clicking and selecting "Run as administrator", enter the 
wsl --install
wsl --list –online //列出可安装的各类wsl版本
wsl --install -d <Distribution Name>//用你想安装的特定版本替代<Distribution Name>

2、在windows中安装合适的驱动WSL requires installation of the following Windows driver.
To install the compatible driver, refer to AMD Software: Adrenalin Edition™ 24.6.1 for WSL 2.
 
3、Install AMD unified driver package repositories and installer script
Enter the following commands to install the installer script for the latest compatible Ubuntu® version:
sudo apt update
wget https://repo.radeon.com/amdgpu-install/6.1.3/ubuntu/jammy/amdgpu-install_6.1.60103-1_all.deb
sudo apt install ./amdgpu-install_6.1.60103-1_all.deb

4、Install AMD unified kernel-mode GPU driver, ROCm, and graphics
Enter the following command to display a list of available usecases:
sudo amdgpu-install --list-usecase
WSL usecase
AMD recommends installing the WSL usecase by default.
Run the following command to install open source graphics and ROCm.
amdgpu-install -y --usecase=wsl,rocm --no-dkms

5、Post-install verification check
Run a post-installation check to verify that the installation is complete:
Check if the GPU is listed as an agent.
rocminfo
  
6、Uninstall ROCm
Run the following command to uninstall the ROCm software stack and other Radeon software for Linux components:
sudo amdgpu-uninstall





LLM solution selection
Popular open source LLM solutions that support AMD GPU/ROCm well can be roughly divided into two categories:

Based on GGML (such as llama.cpp)
llama.cpp should be the most mature open source LLM solution on all consumer platforms. There are countless shell software, which can run from Apple Silicon to AMD APU to various independent graphics cards, and also supports multiple GPUs. Although its performance optimization for GPU is relatively general, it is basically usable.
Example: ./llama-server -t 16 -c 32768 -m ~/models/llama3/llama3-70B/llama3-70B-q8.gguf -ngl 128 -np 16 -cb -fa –host 127.0.0.1 –port 8000 -sm row -mg 1
Based on PyTorch (such as vLLM)
Although vLLM+GPTQ is the official solution on AMD PPT, AMD's development focus is on Instinct, and its Radeon support is far from mature. Currently, only a fork version of AMD can work well on Radeon, and the mainline vLLM does not yet fully support Radeon functions (such as GPTQ support, etc.), so it is currently a future solution.
Example: docker run -it –name=vllm –network=host –group-add=video –ipc=host –cap-add=SYS_PTRACE –security-opt seccomp=unconfined –device /dev/kfd –device /dev/dri/card0 –device /dev/dri/card1 –device /dev/dri/renderD128 –device /dev/dri/renderD129 -v /mnt/models/vllm:/models vllm python -O -u -m vllm.entrypoints.openai.api_server –host=127.0.0.1 –port=8000 –model /models/Meta-Llama-3-70B-Instruct-GPTQ/ –max-model-len 8192
Most other frameworks can also be supported in theory, for example, llama has several onnx ports. However, the graphics card has been in my hands for less than a week, so I can only choose from a few familiar solutions.

The choice of the front end is relatively simple. Use the open-webui of the ollama project to connect to the OpenAI-compatible server of llama.cpp/vLLM. For example, use Docker: docker run -d –network=host -v open-webui:/app/backend/data –name open-webui –restart always ghcr.io/open-webui/open-webui:main

Some pitfalls encountered
The Debian kernel does not enable PCIe P2P support, which makes the multi-GPU of llama.cpp -sm row not work, which is manifested as a crash in cudaMemcpy2DAsync(cudaMemcpyDeviceToDevice).
In the absence of PCIe P2P, llama.cpp multi-GPU can only use the layer partition mode, which cannot play the performance advantage of multi-GPU, but can only play the advantage of video memory capacity. The solution is to change the Ubuntu kernel, or compile the kernel yourself to turn on CONFIG_PCI_P2PDMA and CONFIG_HSA_AMD_P2P.
Running ROCm programs can easily trigger GPU MES errors
The solution is to upgrade to the latest Linux kernel and linux-firmware/amdgpu firmware, or use the officially supported kernel version and dkms kernel module; try not to use the card running calculations to output desktop graphics or run games, you can use the core graphics or a separate power-on card.
vLLM will continue to occupy the GPU even when there is no request, generating about 150w of power consumption per card
vLLM does not always work reliably using the Linux upstream amdgpu kernel module, and using the dkms kernel module can solve many problems.
Although vLLM's Radeon support is still relatively rudimentary, its single-card performance on Radeon GPUs is indeed much stronger than llama.cpp (especially in the weak link of prompt processing mentioned later), so there is some hope.

Performance test
Use llama-bench of the llama.cpp project to perform performance tests on the llama-3 70B q8_0 quantized model, the model size is about 69GB. In actual tests, three GPUs with 24G video memory are just not enough, so at least four 4090/7900XTX or two RTX 6000 Ada/W7900 are required.

Multiple cards are tested using layer split and row split modes respectively.

When using layer split mode, different layers of LLM are scattered on different GPUs. When a single user runs LLM inference, he can only use multiple GPUs in turn and not in parallel. The layer split mode can make full use of the video memory of multiple cards, but the performance of multiple cards cannot be superimposed at all.
When using row split mode, the matrix calculation of LLM is directly distributed on different GPUs. A single user can give full play to the performance advantages of multiple cards. However, this mode has certain requirements for multi-card interconnection. In this generation, both Navi31 and AD102 have cut off XGMI/NVLINK, so PCIe bandwidth will become a major constraint.
The test commands for the two modes are as follows

./llama-bench -m ~/models/llama3/llama3-70B/llama3-70B-q8.gguf -ngl 999 -fa 1 -n 1024 -t 8 -sm row -t 8 -sm layer
./llama-bench -m ~/models/llama3/llama3-70B/llama3-70B-q8.gguf -ngl 999 -fa 1 -n 1024 -t 8 -sm row -t 8 -sm row
Each test group is divided into two results

prompt processing is the step in which LLM processes the text input by the user. The test length is 512 tokens
Reflects the response delay of users in daily use of LLM (especially when the number of conversations reaches a certain length), as well as the throughput of processing large amounts of text work (such as full-text translation, summarization, etc.). The performance bottleneck of prompt processing is generally biased towards CU throughput, especially matrix units.
Text generation 1024 token means generating text, and the test length is 1024 tokens
Reflects the speed of LLM output text. Generally, when running larger models, the main performance bottleneck is the video memory bandwidth.

Test results
It should be noted that under the premise of prompt processing speed of hundreds to thousands of tokens/s, in actual use, even if the conversation context is very long, the difference in prompt processing performance is relatively limited to the user. Therefore, the Radeon solution also has its own characteristics.

Of course, llama.cpp is not the best solution for both GPUs. For example, when W7900 uses vLLM+GPTQ quantization to run llama 3 70B, the single-card text generation can run at about 17 tokens/s, and prompt processing is also significantly faster; NVIDIA also has a faster TensorRT-LLM solution. This is even worse for computing cards: llama.cpp has never been optimized specifically for computing cards, and testing single-user throughput is not the only performance indicator of computing cards. The cost-effectiveness of computing cards running single-user llama.cpp is even worse than that of GDDR graphics cards. 

When doctors use AI, they need to input a large amount of patient information, such as medical history, family history, living habits, etc. Once this information is leaked or misused, it will cause a serious invasion of patient privacy. Therefore, in the process of using generative AI, we must attach great importance to data security and privacy protection to ensure that patients’ personal information is not illegally obtained and used. Currently, most privacy-preserving machine learning solutions based on secure multi-party computation (MPC) run on CPUs, but GPUs have become an indispensable hardware device in the field of large medical models. The powerful computing power of GPU can greatly speed up the training and reasoning of large medical models, especially the calculation speed of neural networks. In order to enable MPC's privacy-preserving machine learning solution to fully utilize the computing performance of GPU and accelerate the training and prediction efficiency of MPC-based privacy-preserving machine learning.
