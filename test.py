from dynamic_reconfiguration_environment import NetEnv 

env=NetEnv()

env.reconfigure()
print(len(env.flows))
env.remove_flow("F10")
print(len(env.flows))