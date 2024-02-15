import os
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
from private_exp2 import AgarwalSinghPrivateEXP2
from privacy_preserving_noise import get_laplace_dist, get_gaussian_dist

def best_fixed_decision(dim, losses):
    best_i, best_val = 0, 1e9
    for i in range(dim):
        cost_i = 0
        for l in losses:
            cost_i += l[i]
        if cost_i < best_val:
            best_i, best_val = i, cost_i
    
    return best_i, best_val


l_fixed = None
def fixed_adversary(x, t):
    return l_fixed


def uniform_adversary(x, t):
    return np.random.uniform(0, 1, x.size)


def best_adversary(x, t):
    return x


def switching_adversary(x, t):
    if t % 2 == 1:
        return l_fixed
    return best_adversary(x, t)


def simulate(dim, T, A, adv):
    cost = 0
    ls = []
    for t in range(1, T + 1):
        i_t = A.predict(t)
        x_t = np.eye(dim)[i_t]
        l_t = adv(x_t, t)
        ls.append(l_t)
        c_t = np.dot(x_t, l_t)
        cost += c_t
        A.observe_loss(t, c_t)
    
    best_i, best_val = best_fixed_decision(dim, ls)
    regret = cost - best_val

    return regret


plot_line_width = 2.5
colors = {0.1: "black", 0.5: "green", 1: "red", 5: "blue", 10: "purple"}
fig = plt.figure()

if __name__ == "__main__":
    adv_type = input("Adversary(Best(1)/Uniform(2)/Fixed(3)/Switching(4): ")
    if adv_type == "1":
        adversary = best_adversary
    elif adv_type == "2":
        adversary = uniform_adversary
    elif adv_type == "3":
        adversary = fixed_adversary
    else:
        adversary = switching_adversary
    
    dim = int(input("Dimension: "))
    l_fixed = np.random.uniform(0, 1, dim)

    Ts = list(np.arange(20, 100, 10)) + list(np.arange(100, 500, 50)) + list(np.arange(500, 2000, 100))
    # Ts = list(np.arange(100, 500, 100)) + list(np.arange(500, 2000, 1000))
    epsilons = [0.1, 0.5, 1, 5, 10]
    delta = 1e-2
    reps = 50
    
    for eps in epsilons:
        regrets = []
        for T in Ts:
            print(f"Calculating for eps={eps} and T={T}...")
            avg_regret = 0

            D_lap = get_laplace_dist(1 / eps)
            print("Laplace: ", 1/eps)

            eta = np.sqrt(np.log2(dim) / (2*dim*T*(1 + 2*np.log2(dim*T)/eps**2)))
            gamma = eta * dim * np.sqrt(1 + 2*np.log2(dim*T)/eps**2)
            mu = np.ones(dim) / dim

            for _ in range(reps):
                A = AgarwalSinghPrivateEXP2(dim, T, D_lap, eta, gamma, mu)
                regret = simulate(dim, T, A, adversary)
                avg_regret += regret
            
            avg_regret /= reps

            regrets.append(avg_regret)
        
        plt.plot(Ts, regrets, linestyle='-', color=colors[eps], linewidth=plot_line_width, label=f'EXP2 eps={eps}')

    plt.ylabel("Average Regret")
    plt.xlabel("T")

    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    os.makedirs("plots", exist_ok=True)
    fig.savefig(os.path.join("plots", f"non_stochastic_multi_armed_bandit_d={dim}.pdf"), bbox_inches='tight')

