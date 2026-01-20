from dynamic_reconfiguration_environment import NetEnv

#Create env
env = NetEnv()

print("\n--- Initial flows ---")
print([f.flow_id for f in env.flows])

#Add 2 flows
env.reconfigure({"add": 2})



print("\nFlows after add:")
print([f.flow_id for f in env.flows])


#Remove 2 flows
to_remove = [env.flows[1].flow_id, env.flows[3].flow_id]
print(f"\n--- Reconfiguration: remove {to_remove} ---")
env.reconfigure({"remove": to_remove})

print("\nFlows after remove:")
print([f.flow_id for f in env.flows])

