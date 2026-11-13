# paper-repro-audit

一个用于核查论文代码、数据和模型开放情况的 AI Skill，可在 Codex 中使用。

## 基本功能

- 检查论文的代码、数据和模型权重是否公开。
- 检查训练、推理和评测流程是否完整。
- 对照论文与代码，分析实现是否一致。
- 评估复现条件，指出缺失材料与需要补充的工作。
- 使用 Python 脚本辅助扫描仓库和追踪调用路径，按固定模板输出中文报告。

## 下载安装

先安装 Node.js 22.20 或更新版本，然后运行以下命令，安装给 Codex。

```bash
npx skills add manifesting696-byte/paper-repro-audit --skill paper-repro-audit -a codex -g
```

也可以在 GitHub 仓库页面点击 **Code → Download ZIP** 下载。

运行辅助脚本需要 Python 3.10+，扫描远程仓库需要 Git。

## 使用方法

在 Codex 中输入以下内容，并附上论文链接或文件。

```text
使用 $paper-repro-audit 检查这篇论文的代码、数据是否公开，
能否复现，以及论文描述与代码实现是否一致。

论文链接……
```

默认进行资料和代码检查，需要实际运行训练或推理时请另外说明。
