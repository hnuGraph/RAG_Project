本项目的文件结构如下:

```markdown
|-- image

​        |-- environment.yml 环境依赖

​        |-- run.sh 执行脚本   

|-- data

​        |-- raw_data

​                |-- 比赛的数据集文件

​        |-- user_data

​                |-- 模型文件（需自己下载）

​                |-- 中间文件、数据、模型权重等

​        |-- prediction_result

​                |-- result.json

​        |-- code

​                |-- 项目代码  

​        |-- README.md  代码说明+项目运行流程介绍

|-- 技术报告

​        |-- 技术报告.word

|-- README.md
```



image文件夹内的environment.yml包含本项目所需要的所有环境，请安装好环境后：



 1. 进入image文件夹，执行 sh run.sh ，可实现训练到推理过程。 进入image文件夹，执行 sh  run2.sh ，可实现用我们微调后的模型推理的过程。
 1. 更加细节的README文件请参考data/README.md，里面包含完整的算法流程和说明。

