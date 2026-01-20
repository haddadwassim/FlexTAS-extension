from dynamic_reconfiguration_environment import NetEnv

env = NetEnv()

print("\n--- Initial flows ---")
print([f.flow_id for f in env.flows])

env.reconfigure({"add": 2})

print("\n--- After adding flows ---")
print([f.flow_id for f in env.flows])

env.reconfigure({"remove": ["F1", "F3"]})

print("\n--- After removing flows ---")
print([f.flow_id for f in env.flows])
 