import csv
import numpy as np
import matplotlib.pyplot as plt


PATH = './results/'
def read(file_name, sliding=60):
    returns = []
    with open(PATH + file_name) as fin:
        fin = csv.reader(fin)
        for r, line in enumerate(fin):
            returns.append([])
            line = list(map(float, line))
            for i in range(0, len(line), sliding):
                returns[r].append(np.mean(line[i:i+sliding]))
    return np.array(returns)

def plot(returns, color='red'):
    std = np.std(returns, axis=0)
    mean = np.mean(returns, axis=0)
    plt.plot(mean, color=color)
    plt.fill_between(np.arange(len(mean)), mean-std, mean+std, color=color, alpha=0.4)


def main():
    norm_obs = []
    for i in range(3):
        norm_obs.append(read('norm_obs{}.csv'.format(i+1)).reshape([-1])  ) 
    # print(norm_obs)
    print(len(norm_obs))
    print(norm_obs[0])
    baseline2 = read('unnorm_obs1.csv')
    # IS = read('IS_returns.csv')
    # IR = read('IR_returns.csv')
    plt.plot(norm_obs[0])
    plt.plot(norm_obs[1])
    plt.plot(norm_obs[2])
    # plt.plot(baseline2, color='orange')
    # plot(IS, color='red')
    # plot(IR, color='green')
    plt.title('Different ways to reuse batch in PPO')
    plt.xlabel('Number of steps')
    plt.ylabel('Episode Return')
    plt.legend(['Baseline1', 'Baseline2'])
    plt.show()


if __name__ == '__main__':
    main()



'''
for e in range(np.shape(returns)[0]):
    run = plt.plot(returns[e, :], color='grey', alpha=0.4)
pid = plt.axhline(-0.36, color='red')
plt.legend([mean[0], between, run[0], pid],
           ["Mean", "Std", "Runs", "PID"])
plt.title('Performance of PPO on DXL Reacher Task')
plt.xlabel('Steps and Walltime')
plt.ylabel('Average Return')

plt.xticks( np.arange(0, 26, 5),["0k/0min", "10k/8min", "20k/16min", "30k/24min", "40k/32min", "50k/40min"])
plt.show()
'''