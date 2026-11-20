import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, BoundaryNorm, rgb2hex
import matplotlib.gridspec as gridspec
from matplotlib.legend_handler import HandlerTuple
import cmcrameri
import cmcrameri.cm as cmc
import pandas as pd
import ptitprince as pt

plt.rcParams['mathtext.fontset'] = 'cm'

def get_short_estimation(window):
    parameters = [
            [-1.0, 1.00, 1.0],
            [-0.9, 1.00, 1.0],
            [-0.8, 1.00, 1.0],
            [-0.7, 1.00, 1.0],
            [-0.6, 1.00, 1.0],
            [-0.5, 1.00, 1.0],
            [-0.4, 1.00, 1.0],
            [-0.3, 1.00, 1.0],
            [-0.2, 1.01, 1.0],
            [-0.1, 1.01, 1.0],
            [0.0, 1.02, 1.0],
            [0.1, 1.03, 1.0],
            [0.2, 1.04, 1.1],
            [0.3, 1.06, 1.2],
            [0.4, 1.09, 1.4],
            [0.5, 1.20, 1.3],
            [0.6, 1.30, 1.9],
            [0.7, 1.50, 3.0],
            [0.8, 2.00, 7.0],
            [0.9, 3.00, 19.0]
        ]
    
    dfs = []
    for i in range(len(parameters)):
        df = pd.read_csv("./short_estimation/time_series_add={:.1f}_inf={:.2f}_R={:.1f}.csv".format(parameters[i][0],parameters[i][1],parameters[i][2]), comment='#')

        for w_name, w_size in window.items():
            df_shift = df.rolling(window=w_size).mean().dropna()

            # parameter a estimate
            add_true = parameters[i][0]
            ab_ob = df_shift.filter(like="ab_ob").to_numpy().mean(axis=1)
            ab_oa = df_shift.filter(like="ab_oa").to_numpy().mean(axis=1)
            inf = parameters[i][1]
            HBH = df_shift.filter(like="HBH").to_numpy().mean(axis=1)
            HBH = inf*HBH
            HAH = df_shift.filter(like="HAH").to_numpy().mean(axis=1)
            a_est_ab_ob = 1.0 - ab_ob/HBH
            a_est_ab_oa = (HAH-ab_oa)/HBH

            # Ruc estimation
            ob_ob = df_shift.filter(like="ob_ob").to_numpy().mean(axis=1)
            oa_ob = df_shift.filter(like="oa_ob").to_numpy().mean(axis=1)
            inf = parameters[i][1]
            HBH = df_shift.filter(like="HBH").to_numpy().mean(axis=1)
            HBH = inf*HBH
            HAH = df_shift.filter(like="HAH").to_numpy().mean(axis=1)

            ruc_est_abob_obob = ob_ob - (1.0-a_est_ab_ob)*(1.0-a_est_ab_ob)*HBH
            ruc_est_aboa_obob = ob_ob - (1.0-a_est_ab_oa)*(1.0-a_est_ab_oa)*HBH
            ruc_est_aboa_oaob = oa_ob + a_est_ab_oa*(1.0-a_est_ab_oa)*HBH

            # format
            d = {
                "add_true":add_true, "window":w_name,
                "a_est_ab_ob":a_est_ab_ob, "a_est_ab_oa":a_est_ab_oa,
                "ruc_est_abob_obob":ruc_est_abob_obob, "ruc_est_aboa_obob":ruc_est_aboa_obob, "ruc_est_aboa_oaob":ruc_est_aboa_oaob
                }
            df_parameter = pd.DataFrame(d)
            dfs.append(df_parameter)

    df_all = pd.concat([df for df in dfs if not df.empty], ignore_index=True)
    return df_all


def raincloud_short_estimation_aest(add_true):
    window = {"1day":4, "1week":4*7, "1month":4*30, "1year":4*365, "10year":4*365*10}

    df_all = get_short_estimation(window)
    df_tmp = df_all.query("{:.2f} < add_true < {:.2f}".format(add_true -0.01, add_true + 0.01))
    print(df_tmp.describe())

    y_set = ["a_est_ab_ob", "a_est_ab_oa"]
    #y_set = ["ruc_est_abob_obob", "ruc_est_aboa_obob", "ruc_est_aboa_oaob"]
    y_set_num = len(y_set)

    fig = plt.figure(figsize = (20, 8))
    #fig.suptitle("No-normalization", fontsize=20)
    for i in range(y_set_num):
        ax = fig.add_subplot(1,y_set_num,i+1)
        ax.axhline(1.0, color="gray")
        ax.axhline(0.0, color="gray")
        ax.axhline(add_true, color="red")
        pt.RainCloud(data=df_tmp, x="window", y=y_set[i], orient="v", ax = ax, palette='Set2', 
                    linewidth = 0.1, width_viol = 1.5, offset=0.1, bw=0.1,  # cloud(violine)
                    rain_alpha = 1.0, move = 0.2,                           # rain(jitter)
                    box_linewidth = 1.5, box_showfliers = False,            # umbrella(box)
                    pointplot = False, point_scale = 0.5                     # thunder(line)
                    )

        ax.set_title(y_set[i],fontsize=20)
        ax.set_xlabel(f"window size", fontsize=20)
        ax.tick_params(axis='x', labelsize=20)
        #ax.set_xlim(-1.0, 1.0)
        ax.set_ylabel(f"est", fontsize=20)
        y_range = [-1, -0.5, 0, 0.5, 1]
        ax.set_yticks(y_range, minor=False)
        ax.set_yticklabels(["{:.1f}".format(n) for n in y_range] ,fontsize=20)
        ax.set_yticks(np.arange(y_range[0], y_range[-1], 0.1), minor=True)
        ax.set_ylim(y_range[0], y_range[-1])

    plt.subplots_adjust(wspace=0.3, hspace=0.3) 
    plt.show()

def raincloud_short_estimation_rucest(add_true):
    window = {"1day":4, "1week":4*7, "1month":4*30, "1year":4*365, "10year":4*365*10}

    df_all = get_short_estimation(window)
    df_tmp = df_all.query("{:.2f} < add_true < {:.2f}".format(add_true -0.01, add_true + 0.01))
    print(df_tmp.describe())

    #y_set = ["a_est_ab_ob", "a_est_ab_oa"]
    y_set = ["ruc_est_abob_obob", "ruc_est_aboa_obob", "ruc_est_aboa_oaob"]
    y_set_num = len(y_set)

    fig = plt.figure(figsize = (20, 8))
    fig.suptitle(f"add={add_true:.1f}", fontsize=20)
    for i in range(y_set_num):
        ax = fig.add_subplot(1,y_set_num,i+1)
        ax.axhline(1.0, color="red")
        ax.axhline(0.0, color="gray")
        pt.RainCloud(data=df_tmp, x="window", y=y_set[i], orient="v", ax = ax, palette='Set2', 
                    linewidth = 0.1, width_viol = 1.5, offset=0.1, bw=0.1,  # cloud(violine)
                    rain_alpha = 1.0, move = 0.2,                           # rain(jitter)
                    box_linewidth = 1.5, box_showfliers = False,            # umbrella(box)
                    pointplot = False, point_scale = 0.5                     # thunder(line)
                    )

        ax.set_title(y_set[i],fontsize=20)
        ax.set_xlabel(f"window size", fontsize=20)
        ax.tick_params(axis='x', labelsize=20)
        #ax.set_xlim(-1.0, 1.0)
        ax.set_ylabel(f"est", fontsize=20)
        y_range = [0.0, 0.5, 1, 1.5, 2.0]
        ax.set_yticks(y_range, minor=False)
        ax.set_yticklabels(["{:.1f}".format(n) for n in y_range] ,fontsize=20)
        ax.set_yticks(np.arange(y_range[0], y_range[-1], 0.1), minor=True)
        ax.set_ylim(y_range[0], y_range[-1])

    plt.subplots_adjust(wspace=0.3, hspace=0.3) 
    plt.show()


if __name__ == "__main__":
    #shift_mean()
    add_true = 0.6
    raincloud_short_estimation_aest(add_true)
    raincloud_short_estimation_rucest(add_true)

    pass
