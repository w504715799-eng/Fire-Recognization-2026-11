# D-Fire原生基线（进行中）

快照UTC：2026-11-05T09:19:41.370016+00:00；已完成44/100轮。配置YOLO26s、seed17、640、batch16、AMP、workers2；fit14122/dev1549，全量覆盖核验通过。

完整阶段CSV在snapshots/development-stage.csv；它是进行中开发集曲线，不是最终test或方法增益。原生fire dev AP选best_fire；resume_last保留优化器仅在云端。

新配对B与W/S/M共同采用原GT COCO fire dev AP选模；与本原生参照的视图/选模差异在总方案披露。两类模型的最终评价统一保留原GT。
