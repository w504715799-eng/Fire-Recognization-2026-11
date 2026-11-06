# 教师CPU审计：2026-11-06

B继续训练，GPU锁与既有队列未改。仅使用现有缓存、64张选定fit的元数据、原8图的区域输入，以及一份现有等权监督。单线程审计约1.013秒，峰值RSS约101MiB；没有加载模型权重或使用GPU。核查前后B均处在已完成20轮的同一时段，不能用这一秒窗口估计精确性能影响。

| 内容 | 原v1实际结果 | 解释 |
|---|---:|---|
| 运行时格式合规 | 7/16（43.75%） | A0/8，B7/8 |
| 只规范语法后的可用回答 | 15/16（93.75%） | 仅移除围栏、单对象数组、给精确枚举补引号；未改答案内容 |
| 全景fire与原标签存在性一致 | 6/8 | 不是独立人工视觉准确率 |
| 全景smoke与原标签存在性一致 | 6/8 | 不是独立人工视觉准确率 |
| 区域fire原类别获支持 | 3/4（其中1条无效） | 有效3条均支持 |
| 区域smoke原类别获支持 | 0/4 | 4条均答fire，需检查视觉语义 |
| 原缓存可用于S的上下文字段 | 0/32 | 原A格式失败；审计修复副本不进入训练 |

缓存文件哈希和原响应不变。非法类别1仍无效，不借GT把它改成fire。limited/unjudgeable/invalid按原规则保持中性；小范围假设W归一化保持4个分组各2图的总权重2，所有权重为正。该诊断不是全fit Qwen训练监督。实际全fit B等权监督14,122图也已核实全1、成员匹配，原fit边界未变。

8组区域输入对照保留在云端artifacts/teacher-audit-20261106/*.jpg，显示v1放大数字与v2后绘细框的同图差异；input-audit.json记录源图SHA、原框ID、坐标、EXIF/显示尺寸和视图哈希。图片不作为新训练样本、不发布到仓库。

independent-review-template.json包含64张fit的128个全景/区域审核条目。reviewer_id、时间、视觉标签与notes均待填写，教师回答另存，不冒称已完成独立人工审核。正式审核先看图和区域再记录可见内容/不确定，随后才与教师对照；不根据审核修改原GT或测试集。

v2_serial、v2_batch4和v2_expanded当前仍待GPU队列执行，报告标记pending。审计程序已支持同8图的图像哈希/框ID/回答对应，以及扩展64图中原8与新增56分开统计。真实语义、GPU吞吐、学生收益均不能由CPU格式检查代替。

后续只读审计命令：
```bash
source cloud/activate.sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' nice -n 10 python -B cloud/audit_teacher_pilots.py --output artifacts/teacher-audit-after-v2
```
这会产生新报告并保留本轮证据，不修改现有GPU队列。

补充助手输入目视审查：微小明亮区域的巨大数字和厚框已在v2消失；烟柱样例两版均能看见主烟柱，因此数字遮挡不足以解释所有仅烟偏fire。记录assistant-input-review.json不充当独立人工审核，原128条人工记录仍待填写。
