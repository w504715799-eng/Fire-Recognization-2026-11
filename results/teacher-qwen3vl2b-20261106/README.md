# Qwen3-VL-2B 与 Qwen3.5-0.8B 完整64图比较

2026-11-06，北京时间。结论：值得开展一次有界的任务LoRA适配候选。2B在原标签火焰存在性上改善明显，但烟的敏感度增加伴随原无烟图上的更多阳性；不把模型更大、格式合法或包含both的区域支持率直接当成教师全面更优。LoRA尚未训练，M→普通融合的GPU队列继续。

两者均使用完整固定64张D-Fire fit开发图、同v2提示与渲染、CPU FP32/6线程/单张推理，各128问。模型系列、处理器及聊天模板不同，因此这不是单独改变参数量的消融。64图已反复用于开发；下表为与原标注的对应诊断，不是独立人工准确率、检测AP或跨域效果。

| 指标 | 0.8B | 2B |
|---|---:|---:|
| 合法输出 | 128/128 | 128/128 |
| 火焰存在性一致 | 50/64 | 60/64 |
| 烟存在性一致 | 49/64 | 47/64 |
| 原有火图中的火阳性 | 18/32 | 29/32 |
| 原无火图中的火阳性 | 0/32 | 1/32 |
| 原有烟图中的烟阳性 | 21/32 | 31/32 |
| 原无烟图中的烟阳性 | 4/32 | 16/32 |
| 火区域支持（fire或both） | 22/32 | 23/32 |
| 烟区域支持（smoke或both） | 22/32 | 31/32 |
| 火区域仅回答fire | 21/32 | 2/32 |
| 烟区域仅回答smoke | 22/32 | 20/32 |
| 区域回答both | 1/64 | 32/64 |
| 区域回答uncertain | 8/64 | 0/64 |
| 同128问纯询问耗时 | 914.19秒 | 1829.01秒 |
| 实测峰值RSS | 5.85GiB | 12.68GiB |

区域both可能包含正确的可见共存内容，但“支持原类”的宽松口径不能证明它能够区分类别。原无烟标签也不保证画面物理上完全没有烟；本轮不补标、不删图、不将差异全部判为教师幻觉。原图的独立语义审查尚未完成。

| 分组 | 0.8B火存在性一致 | 2B火存在性一致 | 0.8B烟存在性一致 | 2B烟存在性一致 |
|---|---:|---:|---:|---:|
| 原8图 | 6/8 | 7/8 | 7/8 | 7/8 |
| 新增56图 | 44/56 | 53/56 | 42/56 | 40/56 |
| 原标注纯火16图 | 8/16 | 13/16 | 12/16 | 3/16 |
| 原标注仅烟16图 | 16/16 | 15/16 | 7/16 | 15/16 |
| 原标注共存16图 | 10/16 | 16/16 | 14/16 | 16/16 |
| 原标注背景16图 | 16/16 | 16/16 | 16/16 | 13/16 |

原8图区域支持：0.8B火2/4、烟3/4；2B火2/4、烟4/4。新增56图区域支持：0.8B火20/28、烟19/28；2B火21/28、烟27/28。四个上下文字段没有原真值，不能以输出变多或更确定当作正确率提高。

## 适配目标与正在进行的对照

适配目标是原标注任务的火/烟存在性与区域类别对齐：保留火焰识别收益，降低烟判断的过度阳性，改善区域类别区分。监督只来自原fit类别/框，原文件不改。沿用train1000图/2000条、dev128图、内部holdout256图，原64图排除在前三者之外；检测器全部14122图fit不变。

既定一次候选为rank8、250更新、640输入、最长90分钟GPU训练预算。当前已完成真实输入2768条预检（69.67秒、0错误、最长460token）、依赖安装及meta结构检查（112个投影、3211264个可训练参数）。真实GPU训练入口仍需接入并通过一步训练的显存/数值验证；本轮没有运行LoRA。

CPU上已启动0.8B和base2B的完整256图/512条短任务留出比较，各4线程、batch4、最长90分钟。两者都用同一套短指令与各自新建语法树；这是任务对照协议，不能与上述v2指标直接相减称适配收益。未来adapter必须与base2B使用相同短协议和同完整holdout，并回查原64图v2上下文退化，才区分提示/任务改变与适配的作用。留出集来自检测器fit，仅是教师内部留出，不称外部泛化。

## 复现与文件

在已授权云端FIRE_R2执行，先激活项目sci-env及教师依赖层：

```bash
source cloud/activate-teacher-runtime.sh
python -B cloud/compare_teacher_candidate.py --output artifacts/teacher-qwen3vl2b/base/comparison64-20261106.json
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python -B cloud/preflight_teacher_instructions.py --cloud-config configs/cloud.experiments.local.json
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python -B cloud/evaluate_teacher_holdout_cpu.py --cloud-config configs/cloud.experiments.local.json --model 0.8b --threads 4 --max-seconds 5400
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python -B cloud/evaluate_teacher_holdout_cpu.py --cloud-config configs/cloud.experiments.local.json --model 2b --threads 4 --max-seconds 5400
```

以上对应已执行步骤；已有结果目录时入口拒绝覆盖，不在正在运行的项目重复启动。私人云配置不上传Git。适配依赖安装由cloud/install_adapter_dependencies.py执行，固定wheel URL/SHA见configs/adapter_dependencies.json，仅改根分区专用层。

comparison-summary.json包含完整分组、混淆计数和成本；artifacts/teacher-qwen3vl2b/base保留全部原回答及逐问时间。export-manifest.json记录压缩前后SHA/大小。M CSV和256图CPU报告是导出时的进行中快照；它们没有完成标记，日志中的上游文档提示也原样保留。既有0.8B失败/止损结果继续保存在先前阶段目录，没有删除失败片段。
