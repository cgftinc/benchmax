import json
from envs.mcp_sandbox import MCPSandbox   # path/module name as in your repo

cfg = json.load(open("mcp_utils/mcp.json"))

# --- create sandbox and auto-discover all tools ---
env = MCPSandbox(cfg)                  # add allowed_tool_list=[...] if you like

# quick smoke-test: list what we’ve got
print(env.get_tool_definitions())

result = env.run_tool("current_time", {"format": ""})
print(result)
result = env.run_tool("current_time", {"format": ""})
print(result)