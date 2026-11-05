# W/S/M与教师开发阶段

2026-11-05：已实现R2原清单适配、全量正权重W、训练语义S、M细节/分工/门控及教师A/B任务、统一原GT评价。三来源四部分全量数据已就绪。

本机59项检查，11项云端后端测试跳过；云端CPU59项全部通过。包含合成训练1轮、保存/重载、原GT dev验证和部署辅助头剥离一致性，测试临时权重已清理，不是模型改进结果。首次端到端测试失败原因及完整日志一并保留；修复后成功。

D-Fire基线仍运行；GPU方法验证与8张固定fit教师先导排队等待正常完成。此阶段没有W/S/M精度增益、教师实际输出或正式test结论。配置为必要对照菜单，不表示已全部训练。

可复现检查：授权云端source cloud/activate.sh，CUDA_VISIBLE_DEVICES为空运行python -B -m unittest discover -s tests -v。GPU阶段见configs/next_stage_dfire.json，使用同一GPU锁并记录失败。
