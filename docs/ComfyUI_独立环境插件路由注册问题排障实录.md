# ComfyUI 独立环境插件路由注册问题排障实录

## 背景

在 `ComfyUI-UniRig` 插件中，我们希望实现 `UniRig: Load Mesh` 节点的动态下拉联动：

- 用户切换 `source_folder`（`input` / `output`）
- 前端请求插件路由
- 返回对应目录下的 mesh 文件列表
- 动态更新 `file_path` 候选项

最初看起来是一个简单的“加一条 HTTP 路由 + 前端 fetch”任务，但在 `comfy-env` 独立环境下，出现了启动失败与 404 并存的问题。

---

## 现象与报错

### 1) 启动阶段直接报错，节点未注册

典型报错：

- `AttributeError: type object 'PromptServer' has no attribute 'instance'`
- 发生位置：模块导入时执行了 `@PromptServer.instance.routes.get(...)`

结果是 custom nodes 导入中断，出现 “Imported 0 nodes”。

### 2) 前端请求 404

前端日志出现：

- `/api/unirig/mesh-files?source_folder=input` 404
- `/api/unirig/mesh-files?source_folder=output` 404

说明前端请求到了，但后端路由未成功注册，或注册路径与请求路径不一致。

---

## 根因分析

根因不是“独立环境无法与主进程通信”，而是 **生命周期时序不一致**：

1. 插件模块会在 metadata scan / 节点扫描阶段被导入。
2. 此时 Web 服务对象可能尚未完全初始化。
3. 在模块顶层直接访问 `PromptServer.instance` 极易失败。
4. 即使后续可用，如果没有再次触发注册逻辑，路由仍然缺失。

一句话：**插件已加载，不代表 `PromptServer.instance` 已就绪**。

---

## 错误示例（不要这样做）

```python
@PromptServer.instance.routes.get("/unirig/mesh-files")
async def handler(request):
    ...
```

这个写法会在模块 import 阶段立即求值，最容易在独立环境/扫描阶段炸掉。

---

## 可行方案设计

目标：

- 不在模块顶层依赖 `PromptServer.instance`
- 在“服务可用时”注册路由
- 对宿主是否自动回调 hook 保持兼容

### 组件 1：`on_custom_loaded(app)` 正式注册

实现一个 hook 函数，直接使用传入的 `app` 注册路由：

- 优先 `app.routes.get(...)`
- 兼容 `app.router.add_get(...)`

并注册两条路径，避免前缀差异：

- `/unirig/mesh-files`
- `/api/unirig/mesh-files`

### 组件 2：根 `__init__.py` 兜底触发（与组件 1 组合使用）

在根 `__init__.py` 中增加 fallback：

- 若 `PromptServer.instance` 已存在，主动调用一次 `on_custom_loaded(instance)`

> 实战结论：`INPUT_TYPES()` 内 best-effort 在该独立环境下不稳定，最终已移除。  
> 当前最终形态是**组合方案**：`on_custom_loaded(app)`（负责注册） + 根 `__init__.py` fallback（负责触发）。

### 最终落地版本（当前代码状态）

1. `nodes/mesh_io.py` 中实现 `on_custom_loaded(app)`，负责真正注册路由。
2. `nodes/__init__.py` 导出该 hook。
3. 根 `__init__.py` 在插件加载后检查 `PromptServer.instance`，就绪则主动调用 `on_custom_loaded`。
4. 不再在 `INPUT_TYPES()` 或节点执行路径中做二次补注册，减少时序噪音和行为分叉。

---

## 关键实现要点

### 1) 路由注册函数（核心思想）

- 保持幂等（已注册则直接返回）
- 捕获异常，避免影响节点导入
- 打印可见日志便于现场判断是否已注册

### 2) 前端请求策略

前端使用原生 `fetch`，并根据实际只请求已注册路径：

- `fetch("/unirig/mesh-files?source_folder=...")`

如果环境可能存在前缀差异，可做双路径兜底。

### 3) 日志策略

建议保留这两类日志：

- “准备/尝试注册”
- “注册成功/失败原因”

这样现场能快速区分：

- hook 没触发
- hook 触发但 app 类型不匹配
- 路由已注册但前端请求错路径

---

## 本次排障后看到的正确信号

当出现以下日志时，说明注册链路已打通：

- `PromptServer instance ready; calling on_custom_loaded fallback`
- `Registered mesh-file routes via on_custom_loaded`
- `[MESH_IO] Registered mesh-file routes via on_custom_loaded`

这表示（按当前最终方案）：

1. `PromptServer` 已就绪；
2. 根 `__init__.py` fallback 生效；
3. mesh-files 路由已成功注册。

---

## 为什么这个问题在独立环境更常见

`comfy-env` 会引入额外的“隔离加载 / 元数据扫描”阶段，模块导入时机更早、更严格。  
很多在单进程“恰好可用”的写法（例如模块顶层直接取单例）会暴露出隐藏时序问题。

因此建议把路由注册视为“延迟副作用”，而不是模块定义的一部分。

---

## 最佳实践总结

1. **不要在模块顶层依赖运行态单例**（如 `PromptServer.instance`）。
2. **用 hook 或延迟触发注册副作用**。
3. **路由注册幂等化**，可重复调用不出错。
4. **前后端路径保持一致并可诊断**（必要时加前缀兼容）。
5. **先保启动稳定，再加动态能力**：排障阶段优先保证节点可加载。

---

## 结语

这次问题本质是“生命周期管理”，不是“功能代码错误”。  
在 ComfyUI 自定义节点（尤其是独立环境）开发中，**时序正确性**和**可回退策略**往往比“写出功能”更重要。

如果你正在为插件加 API、前端联动或异步能力，建议先把“什么时候执行副作用”设计清楚，再动手写业务逻辑，会省掉大量反复排障时间。
