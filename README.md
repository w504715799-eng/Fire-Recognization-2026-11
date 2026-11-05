# Fire-Recognization-2026-11

围绕火焰识别开展离线语义辅助、数据使用优化与火烟协同检测研究。唯一研究方案为[火焰识别研究方案](火焰识别研究方案.md)。研究价值以可复现实验判断，不预先保证期刊录用，也不要求所有模块首次提出。

## 仓库内容

- `src/fire_r2/`、`code/`、`cloud/`：数据准备、划分、W/S/M、教师评分、统一评价、云端运行与恢复代码。
- `configs/`：公开实验和数据版本配置；真实连接配置不提交。
- `references/`：研究核验记录、文献来源及固定划分；不重复发布完整论文、网页缓存或原始数据。
- `artifacts/`：实验步骤、环境版本、检查及传输回执；保留失败和恢复记录。
- `results/`：完整训练指标、实际参数、结果说明和模型校验信息。
- [下载与安装台账](下载与安装台账.md)：资产及安装、下载、清理的唯一记录。

原始数据集、Python环境、缓存、私钥和中间checkpoint不进入Git历史。用户于2026-11-05明确授权的唯一例外是`results/homefire_B_native_seed17_full_20261105/model/best_fire_deploy.pt`，约20.3MB，作为最终小型YOLO26s文件静态保存并同步，SHA见同目录model.json；本机不加载或运行。其他模型权重继续留云端，不把本次许可扩大到大模型或新的权重。中间checkpoint不提交，也不自动删除云端仍用于恢复的文件。

## 当前结果

[Home-fire原版YOLO26s，seed17，100轮](results/homefire_B_native_seed17_full_20261105/README.md)已完成，期间服务器重启后从第6轮恢复。这是单次原版基线的开发验证集结果，不是方法改进、独立测试或跨库结论；完整CSV及参数保留。

研究保留D-Fire、Home-fire、FASDD_CV/UAV三来源四部分，全部148,438张云端原图已核验就绪，版本和映射见`configs/datasets.json`。W/S/M及教师代码通过60项云端CPU检查；D-Fire100轮进行中，GPU方法检查与教师先导已排队，尚无方法增益结论。当前代码与实验进度据实记录，不能把数据准备视为全研究完成。

## 复现与轻量开发

使用`sci-env`；本机只做轻量检查，不下载或运行研究权重，训练在授权云端执行。

```bash
python -B code/restore_repository_metadata.py
python -B -m unittest discover -s tests -v
python -B code/fire_r2_cli.py train
```

首条只恢复仓库内压缩的固定划分元数据，不下载图片或模型。训练入口默认仅生成计划。详见[开发与云端运行](docs/开发与云端运行.md)。各数据和引用代码遵循原来源许可。

## 同步规则

仓库为`https://github.com/w504715799-eng/Fire-Recognization-2026-11.git`，主分支`main`。每个实际研究阶段完成后同步代码、方案、参数、步骤、完整指标及结果说明；进行中结果标注快照时间。推送前运行`python -B code/check_repository_export.py`检查暂存文件，禁止密钥、权重和大数据混入。

保留完整可追溯记录，不通过删除失败实验或中间指标改变结论。远端已有更新先拉取并合并，不强制覆盖。GitHub插件可用于已授权发布；本机Git HTTPS登录与插件登录独立，未配置Git凭据时不把token写进脚本或仓库。
