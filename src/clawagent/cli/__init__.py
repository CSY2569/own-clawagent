"""CLI module — command handlers, display helpers, completers."""

SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/sessions", "列出所有历史会话"),
    ("/load", "加载指定会话（编号来自 /sessions）"),
    ("/new", "创建新会话"),
    ("/model", "切换模型（无参数弹选框，或 ark:doubao-seed-2-0-pro-260215）"),
    ("/models", "列出当前平台可用模型（/models refresh 刷新）"),
    ("/platform", "切换平台（切换后自动弹出模型选框）"),
    ("/temp", "设置 temperature（如 0.7）"),
    ("/max-tokens", "设置最大输出 token 数（如 8192）"),
    ("/compress", "切换压缩策略（trim / token_trim / summarize）"),
    ("/settings", "显示当前设置"),
    ("/rag-search", "直接搜索 RAG 向量库（不经过 LLM）"),
    ("/help", "显示此帮助"),
]
