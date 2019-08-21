#######################################################################
# Copyright (C)                                                       #
# 2019 Baochen Su(subaochen@126.com)                                  #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import numpy as np
import matplotlib.pyplot as ply

WORLD_SIZE = 5
A_POS = [0, 1]
A_PRIME_POS = [4, 1]
B_POS = [0, 3]
B_PRIME_POS = [2, 3]
DISCOUNT = 0.9

# 把动作定义为对x，y坐标的增减改变
# 按照二维数组[x,y]的习惯，grid world坐标的定义为原点在左上角，Y坐标向右延伸，X坐标向下延伸
# 注意到，这和数学上通常的坐标定义是不一样的
ACTIONS = [np.array([0, -1]),  # left
           np.array([-1, 0]),  # up
           np.array([0, 1]),   # right
           np.array([1, 0])]   # down
ACTION_PROB = 0.25


def step(state, action):
    """每次走一步
    :param state:当前状态，坐标的list，比如[1,1]
    :param action:当前采取的动作，是对状态坐标的修正
    :return:下一个状态（坐标的list）和reward
    """
    if state == A_POS:
        return A_PRIME_POS, 10
    if state == B_POS:
        return B_PRIME_POS, 5

    next_state = (np.array(state) + action).tolist()
    x, y = next_state
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        reward = -1.0
        next_state = state
    else:
        reward = 0
    return next_state, reward


def grid_world_value_function():
    """使用iterative policy evaluation算法计算每个单元格的状态价值函数
    """
    # 状态价值函数的初值
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    episode = 0
    history = {}
    while True:
        episode = episode + 1
        # 每一轮迭代都会产生一个new_value，直到new_value和value很接近即收敛为止
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # bellman equation
                    # 由于每个方向只有一个reward和s'的组合，这里的p(s',r|s,a)=1
                    new_value[i, j] += ACTION_PROB * (reward + DISCOUNT * value[next_i, next_j])
        error = np.sum(np.abs(new_value - value))
        history[episode] = error
        if error < 1e-4:
            break
        # 观察每一轮次状态价值函数及其误差的变化情况
        print(f"{episode}-{np.round(error,decimals=5)}:\n{np.round(new_value,decimals=2)}")
        value = new_value
    return history, value


def grid_world_value_function_in_place():
    """使用iterative policy evaluation（in place）算法计算每个单元格的状态价值函数
    :TODO:画出两种类型的算法的收敛速度对比图
    """
    # 状态价值函数的初值
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    episode = 0
    history = {}
    while True:
        episode = episode + 1
        old_value = value.copy()
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                episode_value = 0
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # bellman equation
                    # 由于每个方向只有一个reward和s'的组合，这里的p(s',r|s,a)=1
                    value_s_prime = value[next_i, next_j]
                    episode_value += ACTION_PROB * (reward + DISCOUNT * value_s_prime)
                value[i, j] = episode_value
        error = np.sum(np.abs(old_value - value))
        history[episode] = error
        if error < 1e-4:
            break
        # 观察每一轮次状态价值函数及其误差的变化情况
        print(f"in place-{episode}-{np.round(error,decimals=5)}:\n{np.round(value,decimals=2)}")
    return history, value


def grid_world_optimal_policy():
    """计算格子世界的最优价值函数和最优策略
    """
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    # 通过一个数组来表示每一个格子的最优动作，1表示在相应的方向上最优的
    optimal_policy = np.zeros((WORLD_SIZE, WORLD_SIZE, len(ACTIONS)))
    episode = 0
    while True:
        episode = episode + 1
        # keep iteration until convergence
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                # 保存当前格子所有action下的state value
                action_values = []
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # value iteration
                    action_values.append(reward + DISCOUNT * value[next_i, next_j])
                new_value[i, j] = np.max(action_values)
                optimal_policy[i, j] = get_optimal_actions(action_values)
        error = np.sum(np.abs(new_value - value))
        if error < 1e-4:
            break
        # 观察每一轮次状态价值函数及其误差的变化情况
        print(f"{episode}-{np.round(error,decimals=5)}:\n{np.round(new_value,decimals=2)}")
        value = new_value
    print(f"optimal policy:{optimal_policy}")


def get_optimal_actions(values):
    """计算当前轮次格子的最优动作
    :param values:格子的状态价值
    :return: 当前的最优动作。解读这个最优动作数组，要参考ACTIONS中四个动作的方向定义，
    数值为1表示此动作为最优动作
    """
    optimal_actions = np.zeros(len(ACTIONS))
    indices = np.where(values == np.amax(values))
    for index in indices[0]:
        optimal_actions[index] = 1
    return optimal_actions


def plot_his(history, title):
    for his in history:
        index, error = his.keys(), his.values()
        ply.plot(index, error)
    ply.title(title)
    ply.xlabel("episode")
    ply.ylabel("error")
    if len(history) != 1:
        ply.legend(["two arrays version", "in place version"])
    ply.show()


if __name__ == '__main__':
    history1, _ = grid_world_value_function()
    history2, _ = grid_world_value_function_in_place()
    plot_his([history1, history2], "iterative policy evaluation error")
    grid_world_optimal_policy()
