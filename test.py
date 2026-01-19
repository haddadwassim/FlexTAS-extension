from dynamic_reconfiguration_environment import NetEnv 

env=NetEnv()

print(env.flows[0].flow_id)
env.save_schedule_state("schedule_state.json")
