# 完整教师留出比较与真实LoRA启动

2026-11-06。0.8B与2B均完成同256张内部留出图、512条短任务询问；2B保留原360条回答，两个4线程进程各补76条，合并通过全部查询键、复用回答和输入绑定检查。旧受控中断记录继续保留，最终成功不抹掉原段记录。

| 原标注闭集任务指标 | 0.8B | 2B |
|---|---:|---:|
| 场景火/烟两字段同时一致 | 154/256（60.16%） | 132/256（51.56%） |
| 区域类别一致 | 145/256（56.64%） | 165/256（64.45%） |
| 火存在性原正图阳性 | 85/128 | 117/128 |
| 火存在性原负图阳性 | 2/128 | 36/128 |
| 烟存在性原正图阳性 | 98/128 | 117/128 |
| 烟存在性原负图阳性 | 34/128 | 66/128 |
| 火区域分类正确 | 127/128 | 116/128 |
| 烟区域分类正确 | 18/128 | 49/128 |
| 合法输出 | 512/512 | 512/512 |

配对场景结果：两者都对110张，仅0.8B对44张，仅2B对22张，都错80张。配对区域结果：两者都对133条，仅0.8B对12条，仅2B对32条，都错79条。完整四态分组及混淆计数见artifacts/teacher-heldout-20261106/complete-comparison.json。

2B区域平均一致率增加7.81个百分点，场景两字段联合一致率下降8.59个百分点，不能宣布全面更优。两者均有明显区域fire偏置；2B更敏感，同时有更多原负图阳性。原标签缺少某类不代表画面物理上绝无该类；这里是数据集原标注任务对齐，不是独立人工语义真值、现场误报率或检测器AP。短任务与64图v2的图集和指令均不同，不跨协议相减声称改进。

## 下一步已经执行

cloud/train_teacher_lora.py完成真实2B CPU前向、语义值token损失、反向和一次AdamW更新：8条微批、640输入、梯度累积8、FP32/8线程。总127.579秒，更新118.642秒，loss0.4877119，梯度范数3.8698275，LoRA参数L1变化5.2910，峰值RSS约12.69GiB；基础/视觉冻结，未保存检查adapter。正式运行第一更新与检查的损失及参数变化一致。

根据实测，把**同一个**适配候选安排为云端CPU FP32完整训练：1000图/2000条，rank8、250更新，预计约8.2小时、最多9小时，与M→普通融合GPU训练重叠。配置为configs/qwen3vl2b_lora_cpu_full.json；原GPU候选配置保留历史并标为执行方式已替代，没有同时训练另一轮GPU候选。当前正式LoRA已启动，尚未完成，也没有适配后效果结论。

每25更新保存云端last-adapter及优化器/随机状态，结束保存final-adapter；基础权重不复制，任何adapter/状态.pt均不下载本机或上传Git。云端微型随机结构的保存/重载检查通过，8个adapter张量逐项相同，优化器状态恢复成功；该检查不代替已完成的真实2B一步验证。

已启动configs/next_teacher_lora_eval_20261106.json队列，等待普通融合完整100轮和LoRA完整250更新。随后基础2B与适配2B分别在相同GPU FP16/batch4下执行全部512条短任务及128条冻结v2回归，每阶段最多20分钟。匹配原图视图/提示/模型/adapter SHA；四项上下文没有GT，只记录回归变化，不能把改变称为正确。现有CPU基础教师结果不直接与未来GPU适配结果相减。队列不持GPU锁等待，不抢占另一服务。

## 执行入口与证据

已执行的正式CPU入口：

```bash
source cloud/activate-adapter-runtime.sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false python -B cloud/train_teacher_lora.py --cloud-config configs/cloud.student-controls.local.json --config configs/qwen3vl2b_lora_cpu_full.json --device cpu --full-cpu
```

评价排队入口：

```bash
source cloud/activate-adapter-runtime.sh
python -B cloud/run_next_stage.py --cloud-config configs/cloud.student-controls.local.json --config configs/next_teacher_lora_eval_20261106.json
```

这些任务已在云端运行，不重复启动。私人云配置不发布。export-manifest.json记录17个阶段文本的原始/压缩SHA；正式CPU训练、M和评价队列文件是进行中快照，完整基础教师比较及CPU一步报告才是已完成结果。本机只回收文本，无新依赖或模型下载。
