import os
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
from private_ftrl import AgarwalSinghPrivateFTRL
from privacy_preserving_noise import get_laplace_dist, get_gaussian_dist

def best_fixed_decision(dim, losses):
    l_tot = np.zeros(dim)
    for l in losses:
        l_tot += l
    
    x = cp.Variable(dim)
    objective = cp.Minimize(l_tot @ x)
    a = np.ones(dim)
    constraints = [cp.norm(x) <= 1]
    prob = cp.Problem(objective, constraints)
    obj_value = prob.solve()

    return x.value, obj_value


def oracle_ftrl(L, rate):
    dim = L.size
    x = cp.Variable(dim)
    objective = cp.Minimize(rate * L @ x + cp.sum_squares(x))
    constraints = [cp.norm(x) <= 1]
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve()
    except:
        tmp = np.zeros(dim)
        tmp[0] = 1
        return tmp

    return x.value


def oracle_ftpl(L, rate):
    dim = L.size
    x = cp.Variable(dim)
    objective = cp.Minimize(rate * L @ x)
    constraints = [cp.norm(x) <= 1]
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve()
    except:
        tmp = np.zeros(dim)
        tmp[0] = 1
        return tmp

    return x.value


def best_adversary(x):
    dim = x.size
    l = cp.Variable(dim)
    objective = cp.Maximize(x @ l)
    constraints = [cp.norm(l) <= 1]
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve()
    except:
        return uniform_adversary(x)

    return l.value


def gaussian_adversary(x):
    dim = x.size
    l = np.random.normal(0, 0.4, dim)
    for i in range(dim):
        l[i] = max(-1, min(l[i], 1))
    
    return l


def uniform_adversary(x):
    dim = x.size
    l = np.random.uniform(-1, 1, dim)

    return l


def simulate(dim, T, A, adv):
    cost = 0
    ls = []
    for t in range(1, T + 1):
        x_t = A.predict(t)
        l_t = adv(x_t)
#        print(f"Round {t}: l_t . x_t={np.dot(l_t, x_t)}")
#        print(x_t)
#        print(l_t)
        ls.append(l_t)
        cost += np.dot(x_t, l_t)
        A.observe_loss(t, l_t)
    
    best_x, best_obj = best_fixed_decision(dim, ls)
    regret = cost - best_obj

    return regret


plot_line_width = 2.5
colors = {1: "red", 5: "blue", 10: "purple"}
fig = plt.figure()

if __name__ == "__main__":
    adv_type = input("Adversary(Best(1)/Gaussian(2)/Uniform(3): ")
    if adv_type == "1":
        adversary = best_adversary
    elif adv_type == "2":
        adversary = gaussian_adversary
    else:
        adversary = uniform_adversary
    
    dim = int(input("Dimension: "))
    eta_ftrl = float(input("FTRL Learning Rate: "))

    Ts = list(np.arange(10, 65, 5)) # + list(np.arange(100, 500, 50)) + list(np.arange(500, 1000, 100))
    #Ts = list(np.arange(100, 500, 100)) + list(np.arange(500, 2000, 1000))
    epsilons = [1, 5, 10]
    delta = 1e-2
    reps = 100
    
    for eps in epsilons:
        regrets_ftrl = []
        regrets_ftpl = []
        for T in Ts:
            print(f"Calculating for eps={eps} and T={T}...")
            avg_regret_ftrl = 0
            avg_regret_ftpl = 0

            D_lap = get_laplace_dist(dim * np.log2(T) / eps)
            print("Laplace: ", dim*np.log2(T)/eps)

            std = max(dim * np.sqrt(T / (np.log2(T) * np.sqrt(dim))), (np.sqrt(dim/eps)) * np.log2(T) * np.log2(np.log2(T) / delta))
            D_gauss = get_gaussian_dist(std)
            print("Gaussian: ", std)

            print("")
            for _ in range(reps):
                A = AgarwalSinghPrivateFTRL(dim, T, D_lap, oracle_ftrl, eta_ftrl)
#                print("Simulating FTRL")
                regret_A = simulate(dim, T, A, adversary)
                avg_regret_ftrl += regret_A

                B = AgarwalSinghPrivateFTRL(dim, T, D_gauss, oracle_ftpl, 1)
                regret_B = simulate(dim, T, B, adversary)
                avg_regret_ftpl += regret_B
            
            avg_regret_ftrl /= reps
            avg_regret_ftpl /= reps

            regrets_ftrl.append(avg_regret_ftrl)
            regrets_ftpl.append(avg_regret_ftpl)
        
        plt.plot(Ts, regrets_ftrl, linestyle=(0, (5, 1.8)), color=colors[eps], linewidth=plot_line_width, label=f'FTRL eps={eps}')
        plt.plot(Ts, regrets_ftpl, linestyle='-', color=colors[eps], linewidth=plot_line_width, label=f'FTPL eps={eps}')

    plt.ylabel("Average Regret")
    plt.xlabel("T")

    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    os.makedirs("plots", exist_ok=True)
    fig.savefig(os.path.join("plots", f"olo_over_sphere_d={dim}.pdf"), bbox_inches='tight')

