# AirSim LLM Harness

面向 UE4.27 + AirSim 1.8.1 的本地无人机任务控制平台。大模型只负责目标语义和视觉判断，航线、安全包线、深度定位与飞行控制均由确定性代码完成。

## 当前能力

- Blocks / CityPark 场景白名单与一键托管
- 任务计划预览和人工批准后执行
- 单机 `SimpleFlight` 覆盖搜索、目标居中深度定位、朝向锁定接近、取证和返航
- OpenAI-compatible 多模态适配器和无需密钥的 Mock Provider
- Python 3.7 AirSim JSONL 隔离桥，现代控制面使用 Python 3.11
- React 控制台、实时状态/遥测/事件、暂停、返航、降落、终止和硬停
- SQLite 索引与每次运行独立的可回放工件
- 模型能力探测、连续失败熔断、`SAFE_HOLD`、运行时限与重启恢复
- 每次运行记录 UE/AirSim 版本、源码提交和场景 SHA-256 指纹

## 架构

```text
React Console ── REST/WebSocket ── FastAPI Control Plane
                                      ├─ deterministic planner / safety / state machine
                                      ├─ OpenAI-compatible VLM (vision only)
                                      ├─ SQLite + per-run artifacts
                                      └─ JSONL bridge (Python 3.7) ── AirSim RPC ── UE4.27
```

飞行 Future 在桥的隔离工作进程中等待；JSONL 主进程持续响应遥测、取图与紧急取消。坐标以每次起飞点为局部 NED 原点，兼容 Blocks 与 CityPark 不同的地图原点。

## 快速启动

1. 运行 `scripts/bootstrap.ps1` 创建 `llm-harness` Conda 环境并安装后端、前端依赖。
2. 复制 `.env.example` 为 `.env`；无模型密钥时保持 `HARNESS_PROVIDER=mock`。
3. 运行 `scripts/dev.ps1`。浏览器访问 <http://127.0.0.1:8000>。
4. 选择 `mock` 场景可在不启动 UE 的情况下验证完整 UI/API；选择 Blocks 或 CityPark 会启动真实 UE/AirSim。

控制面在启动场景前调用模型能力探测；探测必须同时通过视觉输入和结构化 `DetectionAssessment` 校验。也可手动调用 `POST /api/provider/probe`。

真实模型配置示例：

```dotenv
HARNESS_PROVIDER=openai-compatible
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL=your-vision-model
LLM_API_KEY=replace-me
```

智谱视觉模型示例：

```dotenv
HARNESS_PROVIDER=openai-compatible
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
LLM_MODEL=glm-5v-turbo
LLM_API_KEY=replace-me
```

密钥只由后端读取，不会进入浏览器、运行清单或模型日志。

## 安全边界

- 监听地址默认仅为 `127.0.0.1`。
- UE 启动路径只能来自 `configs/scenes.json`。
- 模型输出不能产生任意坐标或执行程序，只能提交标准化视觉判断。
- 飞行命令会经过地理围栏、限高、限速、禁飞区和任务状态校验。
- `hard-stop` 仅用于仿真，会停止 UE 进程并把运行标记为失败。
- 低置信度、无深度、跨视角位置不一致不会触发接近；连续三次模型失败进入 `SAFE_HOLD`。
- `configs/environment-baseline.json` 保存安装基线；每次任务的 `manifest.json` 重新计算实际文件指纹。

## 测试

```powershell
conda run -n llm-harness pytest
$env:PATH = "$env:USERPROFILE\anaconda3\envs\llm-harness;$env:PATH"
Push-Location frontend
npm test -- --run
npm run build
Pop-Location
```

真实 Blocks 冒烟测试需要 UE4.27、已编译 AirSim 插件和现有 `airsim` Conda 环境：

```powershell
conda run -n llm-harness python scripts/smoke_blocks.py
```

该脚本经控制面启动 Blocks，等待 AirSim RPC 就绪，再执行不依赖模型的起飞、悬停、RGB/深度取图、降落与场景关闭。

## 外部工程改动

- CityPark 的原始无效 `.uproject` 已备份为 `CityParkEnvironmentCollec.uproject.pre-harness.bak`，修复文件只统一 JSON 和 `AirSim` 模块大小写。
- Fixture 插件源码保存在 `ue-plugins/DroneHarnessFixtures`，部署副本位于 Blocks 与 CityPark 的 `Plugins` 目录；只有带 `-HarnessFixture` 的测试启动才生成目标。
- 全局 AirSim `settings.json` 未修改；所有启动均使用仓库内 `configs/airsim/*.settings.json`。
