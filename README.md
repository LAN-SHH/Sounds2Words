# Sound2Words

Version: 2.0.0

## Version Notes (2.0.0)

本版本完成了以下核心改造：

- 音频处理链路从“固定 3 秒识别”改为“持续采集 + VAD 切段 + 异步转写队列”
- 录音线程、VAD 线程、转写线程解耦，避免转写阻塞录音
- 不再按固定时间切片；改为基于停顿结束切段（长语音有最大段长保护）
- 页面实时转写输出不切片；历史记录与实时展示保持连续文本体验
- 系统音源选择改为弹窗单选列表，支持同名合并与二级 PID 选择
- 系统音频支持 PID 直采，并增加目标 PID 自动纠偏（同应用发声 PID 变化时自动切换）
- GPU 初始化增加自检日志（faster-whisper/ctranslate2 版本、CUDA 设备数、缺失 DLL）
- GPU 失败自动回退 CPU，避免模型加载阶段卡死
- VAD 默认调试输出关闭（终端不再打印 VAD/SEG 明细）
- 导出 DOCX 默认按日期命名，且不写入 APP 名称与版本号

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
- 系统音源选择弹窗（纵向滚动、单选、同名合并、二级 PID）
- 系统输入支持全量采集 / 指定 PID 直采
- 转写文本实时显示（点击开始后开始，点击结束后停止）
- 自动保存防丢（每5秒保存活动会话，异常退出可恢复）
- 历史记录页：按关键词检索、查看会话详情、删除单条记录
- 导出：TXT / SRT / DOCX（DOCX 不包含 APP 名称与版本号头信息）
- 导出命名：按日期命名（例如 `2026-04-06.docx`，重复自动加序号）

## Internal Pipeline

- `audio_capture.py`：持续采集音频帧
- `vad_segmenter.py`：VAD 状态机切段
- `transcriber_worker.py`：异步消费 segment 并转写
- `transcriber.py`：主线程调度与 UI 信号桥接

## GPU Runtime Behavior

- 推理栈：`faster-whisper + ctranslate2`（模型来源 Hugging Face）
- 启动时打印 GPU 自检日志（版本、设备数、缺失 DLL）
- GPU 请求顺序：
  - `device="cuda", compute_type="float16"`
  - 失败后 `device="cuda", compute_type="float32"`
  - 再失败自动回退 CPU
- 当前推荐兼容分支（方案 A）：
  - `CUDA 12 + cuDNN 8 -> ctranslate2==4.4.0`

## Debug Env Switches

- `SOUND2WORDS_DEBUG_FIXED5=1`：绕过 VAD，按固定 5 秒切段（调试录音链路）
- `SOUND2WORDS_VAD_TRACE=1`：开启 VAD 详细日志与帧/段调试 wav
- `SOUND2WORDS_TRACE_AUDIO=1`：开启转写前音频元信息打印
- `SOUND2WORDS_FORCE_CPU=1`：强制 CPU 推理
- `SOUND2WORDS_DISABLE_CUDA=1`：禁用 CUDA 路径

## Mainland China Download Tips

- 应用默认使用 `HF_ENDPOINT=https://hf-mirror.com` 下载模型，并默认关闭 `xet` 通道。
- 若本机已安装 `hf-xet` 且仍走慢链路，建议执行：

```powershell
python -m pip uninstall -y hf-xet
```

- 如需更快下载，可设置 Hugging Face Token：

```powershell
set HF_TOKEN=your_token_here
python -m sound2words
```

