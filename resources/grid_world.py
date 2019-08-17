#######################################################################
# Copyright (C)                                                       #
# 2019 Baochen Su(subaochen@126.com)                                  #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import numpy as np

WORLD_SIZE = 5
A_POS = [0, 1]
A_PRIME_POS = [4, 1]
B_POS = [0, 3]
B_PRIME_POS = [2, 3]
DISCOUNT = 0.9

# 把动作定义为对x，y坐标的增减改变
ACTIONS = [np.array([0, -1]), # up
           np.array([-1, 0]), # left
           np.array([0, 1]),  # down
           np.array([1, 0])]  # right
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
    """计算每个单元格的状态价值函数
    """
    # 状态价值函数的初值
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    episode = 0
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
        if error < 1e-4:
            break
        # 观察每一轮次状态价值函数及其误差的变化情况
        print(f"{episode}-{np.round(error,decimals=5)}:\n{np.round(new_value,decimals=2)}")
        value = new_value


if __name__ == '__main__':
    grid_world_value_function()
