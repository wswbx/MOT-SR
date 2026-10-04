# 复现说明

[English version](REPRODUCIBILITY.md)

## 环境与运行

环境配置见 [environment.yml](environment.yml) 和 [requirements.txt](requirements.txt)。主要版本：Python 3.11.7、PyTorch 2.0.1、Transformers 4.38.1、NumPy 1.26.4、SciPy 1.12.0。

主入口为 [main.py](main.py)。[run_motsr.sh](run_motsr.sh) 启动本地模型服务并运行振荡 1 示例。

```bash
conda env create -f environment.yml
conda activate mot_sr
bash run_motsr.sh
```

## 提示词与数据集

| 任务 | 提示词规范文件 | 数据集 |
| --- | --- | --- |
| 振荡 1 | [specification_oscillator1_numpy.txt](specs/specification_oscillator1_numpy.txt) | [data/oscillator1](data/oscillator1/) |
| 振荡 2 | [specification_oscillator2_numpy.txt](specs/specification_oscillator2_numpy.txt) | [data/oscillator2](data/oscillator2/) |
| 大肠杆菌生长 | [specification_bactgrow_numpy.txt](specs/specification_bactgrow_numpy.txt) | [data/bactgrow](data/bactgrow/) |
| 应力–应变 | [specification_stressstrain_numpy.txt](specs/specification_stressstrain_numpy.txt) | [data/stressstrain](data/stressstrain/) |

方程提示词在 [sampler.py](mot_sr/sampler.py) 和 [buffer.py](mot_sr/buffer.py) 中组装。科学家模块的提示词与工具定义见 [scientist_llm.py](mot_sr/scientist_llm.py) 和 [scientist_defaults.py](mot_sr/scientist_defaults.py)。

## 实验配置

论文中的实验配置如下：

| 配置项 | 设置 |
| --- | --- |
| 骨干模型 | LLaMA-3.1-8B-Instruct 和 GPT-4o-mini |
| 本地部署 | NVIDIA H100 80GB GPU；LLaMA-3.1 采用 4 位量化 |
| 解码参数 | `temperature=0.6`、`top_k=30`、`top_p=0.3` |
| 每次迭代候选方程数 | 4 |
| 迭代次数 | 四个主要基准任务为 2000 次；LSR-Synth-Chemistry 为 1000 次 |

重复运行的随机种子：`42, 43, 44, 45, 46`。

使用 GPT-4o-mini 时，设置 `API_KEY` 环境变量，并向 `main.py` 传入 `--use_api True --api_model gpt-4o-mini`。方程生成器的 API 请求使用 `max_tokens=512`。模型与接口说明见官方[模型文档](https://developers.openai.com/api/docs/models/gpt-4o-mini)和 [Chat Completions API 参考](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)。

## 评估与运行日志

候选方程评估的超时时间为 30 秒。十个系数使用单起点 BFGS 拟合，参数为 `maxiter=500`、`gtol=1e-10`、`eps=1e-12`。方程生成请求失败时自动重试。

[profile.py](mot_sr/profile.py) 将运行时的候选方程日志写入 `<log_path>/samples/`，记录 `sample_order`、`function` 和 `score`，并生成 TensorBoard 统计信息。

## 最终方程

- 四个基准任务的方程结构：[mot_sr_llama31/](results/final_equations/mot_sr_llama31/)。
- 最终 EMRI 修正项：[equation.py](results/final_equations/emri/equation.py)；完整的十维参数向量见 [params.json](results/final_equations/emri/params.json)。

函数接口与使用示例见[最终方程说明](results/final_equations/README.md)。

## EMRI 与自动化

EMRI 数据集由作者生成。方程搜索、候选筛选、帕累托前沿维护和最终方程选择均自动完成。EMRI 修正项的科学解释由两位领域专家共同作者参与完成。
