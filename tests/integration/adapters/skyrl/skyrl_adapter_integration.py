# import pytest
# import ray

# from benchmax.adapters.skyrl.skyrl_adapter import load_benchmax_env_skyrl, RemoteBaseEnvProxy, expose

# # ---- Fixtures and dummies ----
# class DummyActor:
#     """Dummy Ray actor to simulate remote Benchmax service."""
#     def __init__(self):
#         self.calls = []

#     class Method:
#         def __init__(self, name, parent):
#             self.name = name
#             self.parent = parent
#         def remote(self, *args, **kwargs):
#             self.parent.calls.append((self.name, args, kwargs))
#             # return a sentinel value
#             if self.name == 'compute_reward':
#                 return {'a': 1.0, 'b': 2.0}
#             return f"{self.name}-result"

#     @property
#     def list_tools(self):
#         return DummyActor.Method('list_tools', self)

#     @property
#     def compute_reward(self):
#         return DummyActor.Method('compute_reward', self)

#     @property
#     def run_tool(self):
#         return DummyActor.Method('run_tool', self)

#     @property
#     def init_rollout(self):
#         return DummyActor.Method('init_rollout', self)

#     @property
#     def cleanup_rollout(self):
#         return DummyActor.Method('cleanup_rollout', self)

# @pytest.fixture(autouse=True)
# def patch_ray(monkeypatch):
#     dummy = DummyActor()
#     monkeypatch.setattr(ray, 'get_actor', lambda name: dummy)
#     monkeypatch.setattr(ray, 'get', lambda x: x)
#     return dummy

# # ---- Tests for RemoteBaseEnvProxy ----
# def test_list_tools_proxy(patch_ray):
#     proxy = RemoteBaseEnvProxy(actor_name='BenchmaxEnvService', rollout_id='rid')
#     result = proxy.list_tools()
#     assert result == 'list_tools-result'
#     # Ensure no rollout_id injected for list_tools
#     assert patch_ray.calls == [('list_tools', (), {})]


# def test_compute_reward_proxy(patch_ray):
#     proxy = RemoteBaseEnvProxy(actor_name='BenchmaxEnvService', rollout_id='RID123')
#     reward = proxy.compute_reward('task1', 'action1', {'gt': True})
#     assert reward == {'a': 1.0, 'b': 2.0}
#     # rollout_id should be first arg
#     name, args, kwargs = patch_ray.calls[-1]
#     assert name == 'compute_reward'
#     assert args[0] == 'RID123'

# # ---- Tests for _call_tool ----
# class DummyEnv:
#     def __init__(self):
#         self.benchmax_env = RemoteBaseEnvProxy(actor_name='BenchmaxEnvService')
#         self.extras = {}
#     _call_tool = load_benchmax_env_skyrl.__wrapped__.__defaults__[0]._call_tool if False else None

# @pytest.fixture
# def configured_env(patch_ray):
#     # Build a minimal SkyRL env
#     cfg = {}
#     extras = {'init_rollout_args': {}, 'task': 'T1', 'ground_truth': {}}
#     env = load_benchmax_env_skyrl(actor_name='BenchmaxEnvService', env_config=cfg, extras=extras)
#     return env


# def test_call_tool_errors(configured_env):
#     # Not dict
#     assert configured_env._call_tool('not_a_dict') == "Error: Tool command must be a JSON object."
#     # Missing name
#     assert configured_env._call_tool({'arguments': {}}) == "Error: Missing 'name' field in tool command."
#     # Arguments not dict
#     assert "must be a JSON object" in configured_env._call_tool({'name': 'foo', 'arguments': 'bad'})


# def test_call_tool_success_and_truncate(configured_env):
#     # Valid call
#     out = configured_env._call_tool({'name': 'run_tool', 'arguments': {'x': 1}})
#     assert out.startswith('run_tool-result')
#     # Truncate
#     long_res = configured_env._call_tool({'name': 'run_tool', 'arguments': {}}, max_chars=5)
#     assert long_res.endswith('...')

# # ---- Tests for step() ----

# @ pytest.fixture(autouse=True)
# def patch_parse(monkeypatch):
#     # default parse returns no tools
#     monkeypatch.setattr('benchmax.prompts.tools.parse_hermes_tool_call', lambda x: [])


# def test_step_final_reward(configured_env, patch_parse):
#     # parse returns [], so done=True
#     out = configured_env.step('final answer')
#     assert out["done"] is True
#     assert out["reward"] == 3.0
#     assert out["observations"] == []


# def test_step_tool_flow(monkeypatch, configured_env):
#     # parse returns one tool call
#     monkeypatch.setattr('benchmax.prompts.tools.parse_hermes_tool_call', lambda x: [{'name': 'run_tool', 'arguments': {}}])
#     # Stub _call_tool
#     monkeypatch.setattr(configured_env, '_call_tool', lambda call: 'obs-text')
#     out = configured_env.step('<tool_call>{"name": "calculate", "arguments": {"expression": "25 ÷ 5 + 4 × 3"}}</tool_call>')
#     assert out["done"] is False
#     assert out["reward"] == 0.0
#     assert out["observations"] == [{'role': 'user', 'content': 'obs-text'}]


# def test_step_tool_error(monkeypatch, configured_env):
#     monkeypatch.setattr('benchmax.prompts.tools.parse_hermes_tool_call', lambda x: [{'name': 'run_tool', 'arguments': {}}])
#     def bad_call(call):
#         raise RuntimeError('fail')
#     monkeypatch.setattr(configured_env, '_call_tool', bad_call)
#     out = configured_env.step('<tool_call>{"name": "calculate", "arguments": {"expression": "25 ÷ 5 + 4 × 3"}}</tool_call>')
#     print(out)
#     assert out["done"] is False
#     assert out["observations"][0]['content'] == 'fail'

# # ---- Tests for expose decorator ----
# import asyncio
# class FakeEnv:
#     async def foo(self, x):
#         return x * 2

# @pytest.mark.asyncio
# async def test_expose_wrapper():
#     wrapper = expose('foo')
#     class Host:
#         def __init__(self):
#             self.env = FakeEnv()
#     host = Host()
#     res = await wrapper(host, 10)
#     assert res == 20