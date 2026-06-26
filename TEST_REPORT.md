# Clawagent 测试报告

## 单元测试结果矩阵

### `tests/test_agent.py` — Agent (Usage, AgentResponse, _extract_text)

| 测试用例 | 预期 | 实际 | 状态 |
|----------|------|------|------|
| `test_defaults` | Usage() 默认字段为 0 | input_tokens=0, output_tokens=0, cache_read_input_tokens=0 | PASS |
| `test_from_response_metadata` | 从 metadata 正确解析 token 用量 | input=100, output=50, cache_read=20 | PASS |
| `test_from_empty_metadata` | 空 metadata 返回默认 Usage | Usage() | PASS |
| `test_from_metadata_missing_usage_key` | 缺少 usage 键返回默认值 | Usage() | PASS |
| `test_create` | AgentResponse 保存 text 和 usage | text="Hello", usage.input_tokens=10 | PASS |
| `test_plain_string` | 纯字符串直接返回 | "hello" → "hello" | PASS |
| `test_list_of_content_blocks` | 过滤出 text 类型块 | 仅返回 "visible"，忽略 "thinking" | PASS |
| `test_multiple_text_blocks` | 多 text 块合并换行 | ["hello","world"] → "hello\nworld" | PASS |
| `test_empty_list` | 空列表返回空字符串 | [] → "" | PASS |
| `test_none` | None 转为字符串 | None → "None" | PASS |
| `test_no_text_blocks` | 无 text 块返回空字符串 | [{"type":"thinking"}] → "" | PASS |

### `tests/test_config.py` — Config (Settings, PriceConfig, PriceBook, _extract_price)

| 测试用例 | 预期 | 实际 | 状态 |
|----------|------|------|------|
| `test_from_env_with_key` | 设置 API key 后正常初始化 | api_key="sk-test-key", model="deepseek-v4-flash" | PASS |
| `test_from_env_missing_key` | 无 API key 时抛出 ValueError | 正确捕获异常 | PASS |
| `test_custom_model_and_window` | 环境变量覆盖默认值 | model="deepseek-v4-pro", context_window=128000 | PASS |
| `test_invalid_context_window_fallback` | 无效 context_window 使用默认值 | 回退到 1_000_000 | PASS |
| `test_defaults` | PriceConfig 默认全部为 0 | input=0, cache_hit=0, output=0 | PASS |
| `test_custom_values` | PriceConfig 自定义值 | input=1.0, cache_hit=0.02, output=2.0 | PASS |
| `test_empty_book` | 空 PriceBook 返回 PriceConfig() | get("anything") == PriceConfig() | PASS |
| `test_get_known_model` | 查询已知模型返回正确配置 | input_per_1m=5.0 | PASS |
| `test_get_unknown_model` | 查询未知模型返回空 PriceConfig | get("model-b") == PriceConfig() | PASS |
| `test_extract_single` | 从文本中提取价格 | "1.5元 2.0元" → [1.5, 2.0] | PASS |
| `test_extract_empty` | 无匹配返回空列表 | "no prices" → [] | PASS |
| `test_missing_file` | price.txt 不存在返回空 PriceBook | == PriceBook() | PASS |
| `test_parse_table_format` | 正确解析价格表并返回 PriceBook | flash: input=1.0, cache_hit=0.02, output=2.0; pro: input=3.0, cache_hit=0.025, output=6.0 | PASS |

### `tests/test_tools.py` — Tools (get_current_time, greet, read_file, write_file, run_command)

| 测试用例 | 预期 | 实际 | 状态 |
|----------|------|------|------|
| `test_returns_iso_format` | get_current_time 返回 ISO 格式 | datetime.fromisoformat() 不抛异常 | PASS |
| `test_contains_today` | 返回当前年份 | 年份正确 | PASS |
| `test_greets_by_name` | greet 包含名字和 Hello | "Hello" + "Alice" | PASS |
| `test_greets_different_names` | 不同名字均正确 | "Bob" 在结果中 | PASS |
| `test_read_existing_file` | 读取存在的文件 | 返回内容且长度 > 10 | PASS |
| `test_read_nonexistent_file` | 读取不存在的文件 | 返回错误信息 | PASS |
| `test_read_directory` | 读取目录 | 返回目录错误信息 | PASS |
| `test_read_outside_project_fails` | 路径逃逸检测 | 抛出 ValueError("outside the project") | PASS |
| `test_read_project_root_allowed` | 项目根目录可访问 | _resolve_path(".") 返回有效路径 | PASS |
| `test_write_and_read_back` | 写入文件后能正确读回 | 内容 "hello world" 一致 | PASS |
| `test_write_outside_project_fails` | 写入项目外路径 | 抛出 ValueError | PASS |
| `test_write_creates_parent_dirs` | 自动创建上级目录 | 嵌套路径文件写入成功 | PASS |
| `test_echo` | run_command echo 命令 | "hello" 在输出中 | PASS |
| `test_pwd` | run_command pwd | "clawagent" 在输出中 | PASS |
| `test_failing_command` | 失败命令返回退出码 | "exit code" 在结果中 | PASS |
| `test_absolute_path_allowed_if_within` | 绝对路径解析 | 返回绝对路径 | PASS |
| `test_relative_path` | 相对路径解析 | 返回绝对路径 | PASS |

### `tests/test_ui.py` — UI (ConversationStats, _format_*, _context_color)

| 测试用例 | 预期 | 实际 | 状态 |
|----------|------|------|------|
| `test_zero` | 0 秒格式化 | "0s" | PASS |
| `test_seconds` | 45 秒 | "45s" | PASS |
| `test_minutes` | 125 秒 = 2m05s | "2m05s" | PASS |
| `test_exact_minute` | 60 秒 = 1m00s | "1m00s" | PASS |
| `test_negative` | 负数 → 0s | "0s" | PASS |
| `test_under_thousand` | 500 → "500" | "500" | PASS |
| `test_thousands` | 1500 → "1.5K" | "1.5K" | PASS |
| `test_millions` | 3,500,000 → "3.5M" | "3.5M" | PASS |
| `test_zero` | 0 → "0" | "0" | PASS |
| `test_zero` | ¥0.00 | ¥0.00 | PASS |
| `test_small_amount` | 0.005 → ¥0.00（四舍五入） | ¥0.00 | PASS |
| `test_normal_amount` | 1.5 元格式 | 以 ¥ 开头，包含 1.5 | PASS |
| `test_exact_three_decimals` | 0.123 元 | 包含 ¥ | PASS |
| `test_green_below_70` | 0–69% 绿色 | "green" | PASS |
| `test_yellow_70_to_90` | 70–89% 黄色 | "yellow" | PASS |
| `test_red_90_and_above` | 90%+ 红色 | "red" | PASS |
| `test_initial_state` | ConversationStats 初始状态 | total_tokens=0, message_count=0, cost=0 | PASS |
| `test_update_increments` | 单次 update 记录 | cumulative_input=100, message_count=1 | PASS |
| `test_update_accumulates` | 多次 update 累计 | cumulative_input=300, message_count=2 | PASS |
| `test_total_tokens` | total_tokens = input + output | 150 | PASS |
| `test_context_usage_pct` | 上下文使用率计算 | 50000/1000000 = 5.0% | PASS |
| `test_context_usage_zero_window` | 窗口为 0 时返回 0.0 | 0.0 | PASS |
| `test_cost_calculation` | 费用计算 | 2.0 | PASS |
| `test_cost_with_cache_hit` | 缓存命中费用 | 1.01 | PASS |
| `test_elapsed_seconds_monotonic` | 运行时间累加 | > 0 | PASS |
| `test_elapsed_zero_when_no_start_time` | 未设 start_time 时返回 0 | 0.0 | PASS |

## 功能测试结果矩阵

### `tests/test_functional.py` — 工具调用与 Agent 流程

| 测试用例 | 场景 | 预期 | 实际 | 状态 |
|----------|------|------|------|------|
| `test_all_tools_defined` | 验证工具注册 | 5 个工具：get_current_time, greet, read_file, write_file, run_command | 全部注册 | PASS |
| `test_create_agent_returns_compiled_graph` | create_agent 返回可调用图 | 返回 CompiledStateGraph，含 invoke 方法 | 正确 | PASS |
| `test_agent_run_calls_graph_invoke` | agent.run() 调用 graph | 返回 AgentResponse(text="Hello!") | 正确 | PASS |
| `test_read_existing_file` | read_file 读已有文件 | 返回字符串内容，长度 > 10 | 正确 | PASS |
| `test_read_nonexistent` | read_file 读不存在的文件 | 返回错误信息 | 正确 | PASS |
| `test_path_traversal_blocked` | 路径穿越防护 | 抛出 ValueError | 正确 | PASS |
| `test_write_new_file` | write_file 写新文件 | 文件写入成功，内容一致 | 正确 | PASS |
| `test_write_outside_project` | 写入项目外路径 | 抛出 ValueError | 正确 | PASS |
| `test_write_creates_parent_dirs` | 写文件自动创建目录 | 嵌套路径写入成功 | 正确 | PASS |
| `test_echo` | run_command echo | "hello world" 在输出中 | 正确 | PASS |
| `test_failing_command` | run_command 失败命令 | 退出码 42 在结果中 | 正确 | PASS |
| `test_pwd` | run_command pwd | 包含 "clawagent" | 正确 | PASS |
| `test_single_tool_call_flow` | Agent 调工具后返回文本 | 最终消息文本被返回 | 正确 | PASS |
| `test_usage_tracking` | response_metadata 解析 | input=50, output=30, cache_read=10, cache_creation=5 | 正确 | PASS |
| `test_tool_call_with_real_tool` | 真实 write_file 工具执行 | 文件写入，返回大小信息 | 正确 | PASS |
| `test_multi_tool_conversation` | 多步骤对话（含工具调用） | 最终 Agent 返回总结文本 | 正确 | PASS |
| `test_greet_basic` | greet 基本功能 | 包含 "Hello" + "Alice" | 正确 | PASS |
| `test_greet_empty_name` | greet 空名字 | 包含 "Hello" | 正确 | PASS |

### `tests/test_summarizer.py` — 记忆: 会话摘要 (summarizer)

| 测试用例 | 预期 | 实际 | 状态 |
|----------|------|------|------|
| `test_save_and_get` | 保存摘要后可正确读取 | 标题、内容、消息数一致 | PASS |
| `test_get_nonexistent` | 不存在的会话返回 None | None | PASS |
| `test_get_no_db_file` | 数据库文件不存在返回 None | None | PASS |
| `test_update_existing` | 更新现有摘要 | 新标题、新内容、新消息数 | PASS |
| `test_empty` | 空数据库返回空列表 | [] | PASS |
| `test_no_db_file` | 数据库不存在返回空列表 | [] | PASS |
| `test_multiple` | 多个会话摘要 | 全部返回，数量正确 | PASS |
| `test_save_and_load` | 保存消息后可正确读取 | role + content 一致 | PASS |
| `test_load_empty` | 无消息返回空列表 | [] | PASS |
| `test_load_no_db_file` | 数据库不存在返回空列表 | [] | PASS |
| `test_multiple_batches` | 多批保存累积 | 2条正确 | PASS |
| `test_heuristic_fallback` | 无模型时使用启发式 | 返回标题+摘要 | PASS |
| `test_heuristic_empty` | 空文本启发式 | 标题为 "Conversation" | PASS |
| `test_heuristic_counts_lines` | 行数统计 | "3 exchanges" | PASS |
| `test_with_model` | LLM 生成摘要解析 TITLE/SUMMARY | 正确解析标记 | PASS |
| `test_model_fallback_on_error` | LLM 异常回退启发式 | 返回字符串 | PASS |
| `test_no_title_marker` | 无 TITLE/SUMMARY 标记时 | 标题默认 "Conversation"，全文为摘要 | PASS |

### `tests/test_preferences.py` — 记忆: 偏好提取 (preferences)

| 测试用例 | 预期 | 实际 | 状态 |
|----------|------|------|------|
| `test_save_and_load` | 保存偏好后可正确读取 | key/value 一致 | PASS |
| `test_load_empty` | 空数据库返回空列表 | [] | PASS |
| `test_load_no_db_file` | 数据库不存在返回空列表 | [] | PASS |
| `test_limit` | limit 参数限制返回数量 | 最多 3 条 | PASS |
| `test_dedup_by_max_confidence` | 相同 key+value 去重 | 返回 1 条 | PASS |
| `test_multiple_keys` | 不同 key 共存 | 返回 2 条 | PASS |
| `test_different_values_same_key` | 同 key 不同 value 都保留 | 返回 2 条 | PASS |
| `test_chinese_detected` | 中文内容检测 | language=chinese_priority | PASS |
| `test_english_not_detected` | 英文无语言偏好 | 无 language 条目 | PASS |
| `test_no_model` | 无模型时用基础模式 | 提取并持久化 | PASS |
| `test_with_model` | LLM 提取偏好 | 正确解析 JSON 数组 | PASS |
| `test_model_invalid_json` | LLM 返回非 JSON | 空列表 | PASS |
| `test_model_exception` | LLM 异常 | 空列表 | PASS |

### `tests/test_memory_tools.py` — 记忆: 工具接口 (memory_tools)

| 测试用例 | 预期 | 实际 | 状态 |
|----------|------|------|------|
| `test_empty` | 无会话时 list_sessions | "暂无历史会话记录" | PASS |
| `test_with_data` | 有会话时列出 | 包含 s1 和标题 | PASS |
| `test_multiple` | 多个会话全部列出 | 包含 s1 和 s2 | PASS |
| `test_not_found` | 不存在的会话 | "未找到" | PASS |
| `test_summary_only` | recall_session 摘要模式 | 返回标题和内容 | PASS |
| `test_full_recall` | recall_session 完整模式 | 包含 [user] 和 [assistant] | PASS |
| `test_no_session_id` | summarize_session 空 ID | "请提供" | PASS |
| `test_no_messages` | 无消息的会话 | "未找到" | PASS |
| `test_generates_summary` | 生成摘要 | "摘要已生成" | PASS |

## 统计摘要

| 指标 | 数值 |
|------|------|
| 单元测试总数 | 106 |
| 功能测试总数 | 18 |
| **总测试数** | **124** |
| 通过 | 124 |
| 失败 | 0 |
| 测试覆盖率（估算） | > 90% |
| 类型检查 | `mypy src/ tests/` — 0 errors |
| 代码风格 | `ruff check .` — 0 errors |

## 测试覆盖能力

- **模块覆盖**: config, agent, tools, ui, memory (summarizer, preferences, memory_tools)
- **功能覆盖**: 设置加载、价格解析、Agent 创建、消息提取、工具调用（读/写文件、命令执行、时间、问候）、富文本统计面板、会话管理（列表/回忆/摘要）、用户偏好提取
- **安全覆盖**: 路径逃逸防护、项目外写入拦截、空值/异常处理
- **边界覆盖**: 零值、负值、空列表、None、溢出、无效输入
