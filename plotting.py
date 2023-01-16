import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_experiment_1():
    pass

def plot_experiment_2():
    pass

def plot_experiment_3():
    experiment_path = r"images/experiment_3/"

    epsilon = "1e-1"
    df = pd.read_csv(experiment_path + "performance_data_k=3.csv")
    epsilon_df = df[df["epsilon"]==float(epsilon)]
    sns.set_style("white")
    fig = sns.lineplot(data=epsilon_df, x="n_samples", y="execution time", hue="mode")
    fig.set_title(f"Time vs Sample Size: Bloobs\nepsilon {epsilon}, k 3")
    plt.grid()
    plt.savefig(experiment_path + f"time_vs_n_epsilon={epsilon}_k=3.png")
    plt.close()

    df = pd.read_csv(experiment_path + "performance_data_k=15.csv")
    epsilon_df = df[df["epsilon"]==float(epsilon)]
    sns.set_style("white")
    fig = sns.lineplot(data=epsilon_df, x="n_samples", y="execution time", hue="mode")
    fig.set_title(f"Time vs Sample Size: Bloobs\nepsilon {epsilon}, k 15")
    plt.grid()
    plt.savefig(experiment_path + f"time_vs_n_epsilon={epsilon}_k=15.png")
    plt.close()


def plot_experiment_4():
    experiment_path = r"images/experiment_4/"

    # epsilon = "1e-2"
    # df = pd.read_csv(experiment_path + "performance_data.csv")
    # epsilon_df = df[df["epsilon"]==float(epsilon)]
    # sns.set_style("white")
    # fig = sns.lineplot(data=epsilon_df, x="k", y="execution time", hue="mode")
    # fig.set_title(f"Time vs k: Lena and Baboon\nepsilon {epsilon}")
    # plt.grid()
    # plt.savefig(experiment_path + f"time_vs_k_epsilon={epsilon}.png")
    # plt.close()

    df = pd.read_csv(experiment_path + "performance_data_higher_epsilon.csv", index_col=False)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df = df[df["mode"]=="parallel"]
    sns.set_style("white")
    fig = sns.lineplot(data=df, x="epsilon", y="execution time", hue="dataset")
    fig.set_title(f"Time vs epsilon: Lena and Baboon\nk {30}")
    print(df)
    plt.grid()
    plt.savefig(experiment_path + f"time_vs_epsilon_k={30}.png")
    plt.close()
    
def main():
    # plot_experiment_1()
    # plot_experiment_2()
    # plot_experiment_3() 
    plot_experiment_4()

if __name__ == '__main__':
    main()