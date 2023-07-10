from spaceGPT_agent_obf import SpaceGPTAgent

def play_vs_other_agent(env, agent1, agent2, render=True):
    done = False
    obs = env.reset()
    winner = 0
    while not done:
        if render:
            env.render()        
        action = agent1.next_action(obs)
        obs, reward, done, _ = env.step(action)

        if render:
            env.render()        
        if not done:
            next_action = agent2.next_action(obs)
            _, _, done, _ = env.step(next_action)

    env.render()
    winner = env._grid.winner
    agent1Name = agent1.__class__.__name__
    agent2Name = agent2.__class__.__name__
    if agent1Name == agent2Name:
        agent1Name += " 1"
        agent2Name += " 2"
    final_msg = "The winner is " + agent1Name if winner == 1 else "The winner is " + agent2Name if winner == 2 else "It's a draw"
    print(final_msg)

def play_vs_loaded_agent(env, agent, render=True, first=True):
    enemy_agent = load_enemy_agent()
    if first:
        play_vs_other_agent(env, agent, enemy_agent, render)
    else:
        play_vs_other_agent(env, enemy_agent, agent, render)

def load_enemy_agent():
    return SpaceGPTAgent(2, 4)
