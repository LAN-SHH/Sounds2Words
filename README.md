# Sound2Words

Version: 1.0.0

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m sound2words
```

## Features

- 实时输入源：麦克风、系统音频
- 文件输入源：本地音视频文件转写
- 转写文本实时显示（点击开始后开始，点击结束后停止）
- 自动保存防丢（每5秒保存活动会话，异常退出可恢复）
- 历史记录页：按关键词检索、查看会话详情、删除单条记录
- 导出：TXT / SRT / DOCX（Word 仅文件模式）
- 导出命名：按日期命名（例如 `2026-04-06.docx`，重复自动加序号）

## Mainland China Download Tips

- 应用默认使用 `HF_ENDPOINT=https://hf-mirror.com` 下载模型。
- 如需更快下载，可设置 Hugging Face Token：

```powershell
set HF_TOKEN=your_token_here
python -m sound2words
```
