# Profile 工具使用说明

存放 `ncu`、`nsys` 等 GPU profiler 输出文件和脚本。

## 目录结构

```
profile/
├── README.md
├── scripts/ # profile 启动脚本
└── results/ # 报告输出(不进 git)
```


## 常用脚本

```bash
# 看 kernel 详细指标
bash profile/scripts/profile_kernel.sh

# 看性能数据
bash profile/scripts/profile_perf.sh

# 看时间线
bash profile/scripts/profile_timeline.sh
```

## 输出文件位置
所有报告生成在 profile/results/ 下,文件后缀:

- .ncu-rep — ncu 报告
- .nsys-rep — nsys 报告

