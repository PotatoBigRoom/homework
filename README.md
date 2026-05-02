# 材料机器学习模型开发主页

这个目录包含一个材料机器学习项目 homepage 和可运行的 baseline 训练 pipeline，用于展示四项机器学习模型开发任务：

- 晶体稳定性快速预测模型
- 锂离子电池正极材料筛选工具
- 催化活性位点智能识别系统
- 高熵合金相稳定性预测

## 当前运行结果

已经在独立环境 `.venv` 中安装 CUDA 11.8 版 PyTorch，并运行 `models/run_pipeline.py` 生成：

- `results/model_results.json`
- `data/raw/flla_subset.csv`
- `data/raw/wolverton_oxides.csv`
- `data/raw/matbench_glass.csv`
- `results/visualizations/*.png`

主页会自动读取 `results/model_results.json` 展示四个模型的指标和候选材料。

## 怎么跑的

运行命令：

```bash
HTTPS_PROXY=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897 .venv/bin/python models/run_pipeline.py
```

随机种子固定为 `42`。训练脚本执行了：

- `flla`：下载公开结构-形成能数据，抽样 2600 条，用结构/成分描述符训练 PyTorch MLP 形成能 baseline。
- `wolverton_oxides`：下载公开氧化物数据，基于 `e_form/e_hull/gap/formula mass` 构造电压、容量、稳定性代理目标，训练多输出随机森林。
- `ASE`：生成 9 种金属的 fcc(111) 表面和 top/bridge/hollow 位点，训练吸附能代理模型。
- `matbench_glass`：下载金属玻璃形成能力数据，作为合金相形成代理任务训练随机森林分类器。

注意：当前结果是可运行 homepage baseline，不是最终科学模型。电池和催化任务使用代理目标；高熵合金任务使用金属玻璃数据作为相形成代理。若要得到更可信的指标，需要接入 Materials Project API key、真实电池电压数据、DFT 吸附能数据库和高熵合金相图数据。

## 查看方式

直接打开 `index.html`，或在本目录启动一个静态服务：

```bash
python3 -m http.server 8080
```

然后访问 `http://127.0.0.1:8080`。

## 复现训练

```bash
source .venv/bin/activate
HTTPS_PROXY=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897 python models/run_pipeline.py
```

如果公开数据已缓存，代理变量不是必须的。首次下载 matminer 数据集时建议保留代理。

## 文件

- `index.html`：主页结构和内容
- `styles.css`：响应式页面样式
- `script.js`：首页晶体图网络动态背景
- `models/run_pipeline.py`：下载数据、训练四个模型、生成页面结果
- `results/model_results.json`：模型运行结果
- `data/raw/`：下载或导出的原始数据
