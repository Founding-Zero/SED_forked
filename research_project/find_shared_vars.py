from collections import defaultdict
from itertools import combinations

# Define the functions and their variable usage
functions = {
    "anneal": {
        "uses": ["ctx.num_policy_updates_total", "ctx.current_episode", "ctx.tax_frac"],
        "sets": ["ctx.tax_frac", "ctx.optimizer"],
    },
    "collect": {
        "uses": [
            "ctx.episode_step",
            "ctx.next_obs",
            "ctx.next_done",
            "ctx.principal_next_obs",
            "ctx.principal_next_done",
            "ctx.agent",
            "ctx.principal_agent",
            "ctx.tax_values",
            "ctx.principal",
            "ctx.num_envs",
            "ctx.num_agents",
            "ctx.device",
            "ctx.selfishness",
            "ctx.trust",
            'ctx.principal_tensordict["cumulative_rewards"]',
            "ctx.principal.objective",
            "ctx.tax_frac",
            "ctx.episode_step",
            "ctx.principal_episode_rewards",
            "ctx.episode_rewards",
            "ctx.episode_world_obs",
            "ctx.num_updates_for_this_ep",
        ],
        "sets": [
            "ctx.agent_tensordict",
            "ctx.principal_tensordict",
            "ctx.tax_values",
            "ctx.next_obs",
            "ctx.next_done",
            "ctx.principal_next_obs",
            "ctx.principal_next_done",
            'ctx.principal_tensordict["cumulative_rewards"]',
            "ctx.episode_step",
            "ctx.principal_episode_rewards",
            "ctx.episode_rewards",
            "ctx.episode_world_obs",
        ],
    },
    "save": {
        "uses": [
            "ctx.current_episode",
            "ctx.num_updates_for_this_ep",
            "ctx.agent_tensordict",
        ],
        "sets": [""],
    },
    "policy": {
        "uses": [
            "ctx.agent",
            "ctx.next_obs",
            "ctx.next_done",
            "ctx.device",
            "ctx.agent_tensordict",
            "ctx.principal_agent",
            "ctx.principal_next_obs",
            "ctx.next_cumulative_reward",
            "ctx.principal_next_done",
            "ctx.principal_tensordict",
            "ctx.principal_advantages",
            "ctx.b_returns",
            "ctx.b_values",
            "ctx.optimizer",
            "ctx.num_updates_for_this_ep",
        ],
        "sets": ["ctx.principal_advantages", "ctx.b_returns", "ctx.b_values"],
    },
    "principal": {
        "uses": [
            "ctx.principal_tensordict",
            'ctx.principal_tensordict["cumulative_rewards"]',
            "ctx.num_agents",
            "ctx.principal_advantages",
            "ctx.principal_returns",
            "ctx.principal_agent",
            "ctx.principal_optimizer",
            "ctx.b_values",
            "ctx.b_returns",
            "ctx.num_updates_for_this_ep",
            "ctx.current_episode",
        ],
        "sets": [
            "ctx.principal_advantages",
            "ctx.b_returns",
            "ctx.b_values",
            "ctx.num_updates_for_this_ep",
        ],
    },
    "log": {
        "uses": [
            "ctx.current_episode",
            "ctx.episode_world_obs",
            "ctx.optimizer",
            "ctx.episode_rewards",
            "ctx.num_policy_updates_per_ep",
            "ctx.principal_episode_rewards",
            "ctx.num_agents",
            "ctx.tax_values",
            "ctx.agent",
            "ctx.principal_agent",
        ],
        "sets": [""],
    },
}

variable_usage = defaultdict(lambda: {"used_by": set(), "set_by": set()})
for func, vars in functions.items():
    for var in vars["uses"]:
        variable_usage[var]["used_by"].add(func)
    for var in vars["sets"]:
        variable_usage[var]["set_by"].add(func)

# Count overlap for each variable
overlap_counts = defaultdict(int)
for var, funcs in variable_usage.items():
    overlap_counts[var] = len(funcs["used_by"]) + len(funcs["set_by"])

# Sort variables by overlap count
sorted_overlap = sorted(overlap_counts.items(), key=lambda x: x[1], reverse=True)

# Print the variables with the most overlap and their associated functions
print("Variables with the most overlap:")
for var, count in sorted_overlap:
    print(f"{var}: {count} functions")
    print(f"  Used by: {variable_usage[var]['used_by']}")
    print(f"  Set by: {variable_usage[var]['set_by']}")

# Optionally, also print which functions each variable is associated with
print("\nDetailed variable usage:")
for var, funcs in variable_usage.items():
    print(f"{var} is used by {funcs['used_by']} and set by {funcs['set_by']}")
