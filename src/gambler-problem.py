import numpy as np
import matplotlib.pyplot as plt


def value_iteration_for_gamblers(p_h, theta=0.0001, gamma=1.0):
    """
    Args:
        p_h: Probability of the coin coming up heads
        theta: 迭代结束条件
        gamma: 衰减因子
    """
    # The reward is zero on all transitions except those on which the gambler reaches his goal,
    # when it is +1.
    R = np.zeros(101)
    R[100] = 1

    # We introduce two dummy states corresponding to termination with capital of 0 and 100
    V = np.zeros(101)

    def one_step_lookahead(s, V, R):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            s: The gambler’s capital. Integer.当前的状态
            V: The vector that contains values at each state.
            R: The reward vector.

        Returns:
            A vector containing the expected value of each action.
            Its length equals to the number of actions.
        """
        A = np.zeros(101)
        # Your minimum bet is 1, maximum bet is min(s, 100-s).
        stakes = range(1, min(s, 100 - s) + 1)
        for a in stakes:
            # R[s+a], R[s-a] are immediate rewards.
            # V[s+a], V[s-a] are values of the next states.
            # This is the core of the Bellman equation: The expected value of your action is
            # the sum of immediate rewards and the value of the next state.
            A[a] = p_h * (R[s + a] + V[s + a] * gamma) + (1 - p_h) * (
                    R[s - a] + V[s - a] * gamma)
        return A

    # 第几次重新设置状态价值函数？
    sweep = 0
    while True:
        print("=============sweep=" + str(sweep) + "==============")
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(1, 100):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V, R)
            # print(s,A,V) if you want to debug.
            # 大量的调试输出
            # print("s=" + str(s))
            # print("A=")
            # print(A)
            # print("V=")
            # print(V)
            # print("R=")
            # print(R)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function. Ref: Sutton book eq. 4.10.
            V[s] = best_action_value

        sweep = sweep + 1
        print(f"sweep={sweep},V=")
        print(V)

        # 画出每次迭代的状态价值函数曲线，观察状态价值函数的变化趋势
        plt.plot(range(100), V[:100])

        # Check if we can stop
        if delta < theta:
            plt.xlabel('Capital')
            plt.ylabel('Value Estimates')
            plt.title('Final Policy(action stakes) vs. State(Capital),p_h=' + str(p_h))
            # plt.show()

            break

    # Create a deterministic policy using the optimal value function
    policy = np.zeros(100)
    for s in range(1, 100):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V, R)
        # print("caculate policy,s={:2d},A=".format(s))
        # print(A)
        # how many best actions? and what are they?
        best_action = np.argmax(A)
        # print_best_actions(s, A)
        # Always take the best action
        policy[s] = best_action

    return policy, V


def print_best_actions(s, A):
    counter = 0
    maxItem = max(A)
    index = np.argmax(A)
    for a in A:
        if a == maxItem:
            print("s=" + str(s) + ", best action=" + str(a))
            counter = counter + 1
    print(f"s={s},counter={counter},stake={index}")


def draw_value_estimates(p_h):
    x = range(100)
    y = v[:100]
    plt.plot(x, y)
    plt.xlabel('Capital')
    plt.ylabel('Value Estimates')
    plt.title('Final Policy (action stake) vs State (Capital),p_h=' + str(p_h))
    plt.show()


def draw_policy(p_h):
    x = range(100)
    y = policy
    plt.bar(x, y, align='center', alpha=0.5)
    # plt.plot(x, y)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')
    plt.title('Capital vs Final Policy(ph=' + str(p_h) + ')')
    plt.show()

def draw_multi_policy(policies):
    i = 1
    for policy in policies:
        ax = plt.subplot(330+i)
        plt.plot(range(0,100),policy)
        plt.title(f'P(win)={i/10}',pad=-20,loc='left')
        i+=1
    plt.show()


if __name__ == '__main__':
    policies = []
    for p_h in np.arange(0.1,1,0.1):
        policy, v = value_iteration_for_gamblers(p_h)
        policies.append(policy)

        print("Optimized Policy(p_h=" + str(p_h) + "):")
        print(policy)
        print("")

        print("Optimized Value Function(p_h=" + str(p_h) + "):")
        print(v)
        print("")

        # draw_value_estimates(p_h)
        # draw_policy(p_h)
    draw_multi_policy(policies)
