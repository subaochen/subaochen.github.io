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
        stakes = range(1, min(s, 100 - s) + 1)  # Your minimum bet is 1, maximum bet is min(s, 100-s).
        for a in stakes:
            # R[s+a], R[s-a] are immediate rewards.
            # V[s+a], V[s-a] are values of the next states.
            # This is the core of the Bellman equation: The expected value of your action is
            # the sum of immediate rewards and the value of the next state.
            A[a] = p_h * (R[s + a] + V[s + a] * gamma) + (1 - p_h) * (
                    R[s - a] + V[s - a] * gamma)
        return A

    while True:
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(1, 100):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V, R)
            # print(s,A,V) if you want to debug.
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function. Ref: Sutton book eq. 4.10.
            V[s] = best_action_value

        # 画出每次迭代的状态价值函数曲线，观察状态价值函数的变化趋势
        plt.plot(range(100), V[:100])

        # Check if we can stop
        if delta < theta:
            plt.xlabel('Capital')
            plt.ylabel('Value Estimates')
            plt.title('Final Policy(action stakes) vs. State(Capital),p_h=' + str(p_h))
            plt.show()

            break

    # Create a deterministic policy using the optimal value function
    policy = np.zeros(100)
    for s in range(1, 100):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V, R)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s] = best_action

    return policy, V


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
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')
    plt.title('Capital vs Final Policy(ph=' + str(p_h) + ')')
    plt.show()


if __name__ == '__main__':
    for p_h in (0.35,):
        policy, v = value_iteration_for_gamblers(p_h)

        print("Optimized Policy:")
        print(policy)
        print("")

        print("Optimized Value Function:")
        print(v)
        print("")

        # draw_value_estimates(p_h)
        draw_policy(p_h)
