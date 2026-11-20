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
import multiprocessing
import glob

plt.rcParams['mathtext.fontset'] = 'cm'

def get_optimal(df):
    df_calc = df.query('rmse_a > 0.0')
    df_calc["RMSEratio"] = df_calc["rmse_a"] / df_calc["rmse_o"]

    return df_calc.loc[df_calc["rmse_a"].idxmin()]

def check_estimated_parameters():
    fig = plt.figure(figsize = (18, 10))
        
    ax1 = fig.add_subplot(1,2,1)
    ax1.set_title("(a) Point estimation : $a^{est}_i$", fontsize=20)
    ax1.set_xlabel("Location", fontsize=20)
    x_major_tick = [1, 10, 20, 30, 40]
    x_minor_tick = np.arange(1,40)
    ax1.set_xticks(x_major_tick,minor=False)
    ax1.set_xticklabels(["{}".format(x_major_tick[i]) for i in range(len(x_major_tick))],fontsize=20)
    ax1.set_xticks(x_minor_tick,minor=True)
    ax1.set_xlim(0.0,41.0)
    ax1.set_ylabel("Estimated parameter $a^{est}_i$", fontsize=20)
    y_major_tick = [-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1.0]
    y_minor_tick = np.arange(-1.0,1.0,0.1)
    ax1.set_yticks(y_major_tick,minor=False)
    ax1.set_yticklabels(["{}".format(y_major_tick[i]) for i in range(len(y_major_tick))],fontsize=20)
    ax1.set_yticks(y_minor_tick,minor=True)
    ax1.set_ylim(-0.1,1.0)
    ax1.hlines(0.0, xmin=0,xmax=41, colors='black', linewidth=0.8)
    ax1.hlines(0.6, xmin=0,xmax=41, colors='black', linewidth=0.8)
    ax1.vlines(10, ymin=-1,ymax=2, colors="black", linewidth=0.8)
    ax1.vlines(30, ymin=-1,ymax=2, colors="black", linewidth=0.8)

    ax2 = fig.add_subplot(1,2,2)
    ax2.set_title("(b) Point estimation : $r^{uc\_est}_i$", fontsize=20)
    ax2.set_xlabel("Location", fontsize=20)
    x_major_tick = [1, 10, 20, 30, 40]
    x_minor_tick = np.arange(1,40)
    ax2.set_xticks(x_major_tick,minor=False)
    ax2.set_xticklabels(["{}".format(x_major_tick[i]) for i in range(len(x_major_tick))],fontsize=20)
    ax2.set_xticks(x_minor_tick,minor=True)
    ax2.set_xlim(0.0,41.0)
    ax2.set_ylabel("Estimated parameter $r^{uc\_est}_i$", fontsize=20)
    y_major_tick = [0.8, 0.9, 1.0, 1.1, 1.2]
    y_minor_tick = np.arange(0.0,2.0,0.1)
    ax2.set_yticks(y_major_tick,minor=False)
    ax2.set_yticklabels(["{}".format(y_major_tick[i]) for i in range(len(y_major_tick))],fontsize=20)
    ax2.set_yticks(y_minor_tick,minor=True)
    ax2.set_ylim(0.8,1.2)
    ax2.hlines(1.0, xmin=0,xmax=41, colors='black', linewidth=0.8)
    ax2.vlines(10, ymin=-1,ymax=2, colors="black", linewidth=0.8)
    ax2.vlines(30, ymin=-1,ymax=2, colors="black", linewidth=0.8)

    num = [1, 10, 30, 39]
    name = ["$i=1$", "$i=1, \cdots, 10$", "$i=1, \cdots, 30$", "$i=1, \cdots, 39$"]
    #name = ["$a^{true}_i=0.6 (i=1)$\n$a^{true}_i=0.0 (i=2, \cdots, 40)$",
    #        "$a^{true}_i=0.6 (i=1, \cdots, 10)$\n$a^{true}_i=0.0 (i=11, \cdots, 40)$",
    #        "$a^{true}_i=0.6 (i=1, \cdots, 30)$\n$a^{true}_i=0.0 (i=31, \cdots, 40)$",
    #        "$a^{true}_i=0.6 (i=1, \cdots, 39)$\n$a^{true}_i=0.0 (i=40)$",]
    for i in range(len(num)):
        df = pd.read_csv(f"./point_estimation/point_estimation_{num[i]}point_add=0.6.csv", comment='#')
        df_optimal = get_optimal(df)
    
        HBH = df_optimal.filter(like="HBH").to_numpy()
        inf = df_optimal["multi_inf"]
        HBH = inf * HBH
        ob_ob = df_optimal.filter(like="ob_ob").to_numpy()
        ab_ob = df_optimal.filter(like="ab_ob").to_numpy()
    
        a_est = 1.0 - ab_ob/HBH
        ruc_est = ob_ob - ab_ob*ab_ob/HBH
    
        ax1.plot(np.arange(1,41), a_est, "-o", label=name[i], zorder=40-num[i])
        ax2.plot(np.arange(1,41), ruc_est, "-o", label=name[i], zorder=40-num[i])

    ax1.legend(title="$a^{true}_i=0.6$ location", title_fontsize=20, loc="upper right", fontsize=18)
    ax2.legend(title="$a^{true}_i=0.6$ location", title_fontsize=20, loc="upper right", fontsize=18)
    #ax1.legend(loc="upper right", fontsize=17, ncols=2)
    #ax2.legend(loc="upper right", fontsize=17, ncols=2)
    fig.subplots_adjust(bottom=0.07,top=0.95,left=0.08,right=0.98)
    plt.show()

def output_estimated_parameters(add, num):
    for i in range(len(num)):
        df = pd.read_csv(f"./point_estimation/point_estimation_{num[i]}point_add=0.6.csv", comment='#')
        df_optimal = get_optimal(df)
    
        HBH = df_optimal.filter(like="HBH").to_numpy()
        inf = df_optimal["multi_inf"]
        HBH = inf * HBH
        ob_ob = df_optimal.filter(like="ob_ob").to_numpy()
        ab_ob = df_optimal.filter(like="ab_ob").to_numpy()
    
        a_est = 1.0 - ab_ob/HBH
        ruc_est = ob_ob - ab_ob*ab_ob/HBH

        df_est = pd.DataFrame(np.array([a_est, ruc_est]).T, columns=["add_est", "ruc_est"])
        #print(df_est)
        df_est.to_csv(f"./point_estimation/estimated_parameters_{num[i]}point_add={add:.1f}.csv", index=False)

def show_optimal_ETKF_and_ETKFCC(add, num):
    print("\n##### point estimatiion : optimal ETKF #####")
    n = np.empty(len(num), dtype=int)
    inf = np.empty(len(num), dtype=float)
    r = np.empty(len(num), dtype=float)
    rmse_ratio = np.empty(len(num), dtype=float)
    rmse_a = np.empty(len(num), dtype=float)
    rmse_f = np.empty(len(num), dtype=float)
    rmse_o = np.empty(len(num), dtype=float)
    for i in range(len(num)):
        df = pd.read_csv(f"./point_estimation/point_estimation_{num[i]}point_add={add:.1f}.csv", comment='#')
        df_optimal = get_optimal(df)
        n[i] = num[i]
        inf[i] = df_optimal["multi_inf"]
        r[i] = df_optimal["r"]
        rmse_ratio[i] = df_optimal["RMSEratio"]
        rmse_a[i] = df_optimal["rmse_a"]
        rmse_f[i] = df_optimal["rmse_f"]
        rmse_o[i] = df_optimal["rmse_o"]
    df_etkf = pd.DataFrame(np.array([n, inf, r, rmse_ratio, rmse_a, rmse_f, rmse_o]).T, columns=["n", "inf", "r", "RMSEratio", "rmse_a", "rmse_f", "rmse_o"])
    print(df_etkf)

    print("\n##### point estimatiion : optimal ETKFCC with true #####")
    n = np.empty(len(num), dtype=int)
    inf = np.empty(len(num), dtype=float)
    rmse_ratio = np.empty(len(num), dtype=float)
    rmse_a = np.empty(len(num), dtype=float)
    rmse_f = np.empty(len(num), dtype=float)
    rmse_o = np.empty(len(num), dtype=float)
    for i in range(len(num)):
        df = pd.read_csv(f"./point_estimation_cc/etkfcc_with_true_{num[i]}point_add={add:.1f}.csv", comment='#')
        df_optimal = get_optimal(df)
        n[i] = num[i]
        inf[i] = df_optimal["multi_inf"]
        rmse_ratio[i] = df_optimal["RMSEratio"]
        rmse_a[i] = df_optimal["rmse_a"]
        rmse_f[i] = df_optimal["rmse_f"]
        rmse_o[i] = df_optimal["rmse_o"]
    df_etkfcc_with_true = pd.DataFrame(np.array([n, inf, rmse_ratio, rmse_a, rmse_f, rmse_o]).T, columns=["n", "inf", "RMSEratio", "rmse_a", "rmse_f", "rmse_o"])
    print(df_etkfcc_with_true)

    print("\n##### point estimatiion : optimal ETKFCC with true #####")
    n = np.empty(len(num), dtype=int)
    inf = np.empty(len(num), dtype=float)
    rmse_ratio = np.empty(len(num), dtype=float)
    rmse_a = np.empty(len(num), dtype=float)
    rmse_f = np.empty(len(num), dtype=float)
    rmse_o = np.empty(len(num), dtype=float)
    for i in range(len(num)):
        df = pd.read_csv(f"./point_estimation_cc/etkfcc_with_est_{num[i]}point_add={add:.1f}.csv", comment='#')
        df_optimal = get_optimal(df)
        n[i] = num[i]
        inf[i] = df_optimal["multi_inf"]
        rmse_ratio[i] = df_optimal["RMSEratio"]
        rmse_a[i] = df_optimal["rmse_a"]
        rmse_f[i] = df_optimal["rmse_f"]
        rmse_o[i] = df_optimal["rmse_o"]
    df_etkfcc_with_est = pd.DataFrame(np.array([n, inf, rmse_ratio, rmse_a, rmse_f, rmse_o]).T, columns=["n", "inf", "RMSEratio", "rmse_a", "rmse_f", "rmse_o"])
    print(df_etkfcc_with_est)


### p test ###
def bootstrap_RMSE_ratio_unit(df):
    df_sample = df.sample(n=df.shape[0], replace=True)
    RMSE_ratio = df_sample.mean().to_numpy()
    #print(RMSE_ratio)
    return RMSE_ratio

def bootstrap_RMSE_ratio():
    print("######  sampling RMSE-ratio with bootstrap (point estimation)  ######")
    sample_num = int(1e+05)

    add = 0.6
    ns = [1, 10, 30, 39]

    data = np.empty([sample_num,0])
    line = ""

    for i in range(len(ns)):
        print(f"n={ns[i]}")
        optimal_etkf_time_series = pd.read_csv(f"./point_estimation/optimal_timeseries_etkf_{ns[i]}point_add={add:.1f}.csv", comment='#')
        optimal_etkfcc_with_true_time_series = pd.read_csv(f"./point_estimation_cc/optimal_timeseries_etkfcc_with_true_{ns[i]}point_add={add:.1f}.csv", comment='#')
        optimal_etkfcc_with_est_time_series = pd.read_csv(f"./point_estimation_cc/optimal_timeseries_etkfcc_with_est_{ns[i]}point_add={add:.1f}.csv", comment='#')

        time_series = pd.concat([optimal_etkf_time_series["rmse_ratio"], optimal_etkfcc_with_true_time_series["rmse_ratio"], optimal_etkfcc_with_est_time_series["rmse_ratio"]],axis=1)

        with multiprocessing.Pool(processes=16) as pool:
            result = pool.map(bootstrap_RMSE_ratio_unit,[time_series for i in range(sample_num)])
        data = np.hstack([data,result])
        line = line + f"opt_ETKF_n={ns[i]}_add={add:.1f},opt_ETKFCC_with_true_n={ns[i]}_add={add:.1f},opt_ETKFCC_with_est_n={ns[i]}_add={add:.1f},"
    
    line = line.strip(",") + "\n"

    file = open("./RMSE_ratio_sample_point_est.csv", 'w')
    file.writelines("#sample_size={}\n".format(sample_num))
    file.writelines(line)
    np.savetxt(file, data, delimiter=',')
    file.close()
        
def p_test_RMSE_ratio():
    print("########### RMSE ratio #################")
    df = pd.read_csv('./RMSE_ratio_sample_point_est.csv', comment='#')
    sample_num = df.shape[0]
    add = 0.6
    ns = [1, 10, 30, 39]

    print("ETKF V.S. ETKFCC with true")
    
    for i in range(len(ns)):
        diff =  df[f"opt_ETKFCC_with_true_n={ns[i]}_add={add:.1f}"].to_numpy() - df[f"opt_ETKF_n={ns[i]}_add={add:.1f}"].to_numpy()
        positive = np.count_nonzero(diff > 0)
        negative = np.count_nonzero(diff <= 0)
        print("n={}, add={:.1f} : p-value={}   (positive={}, check={})".format(ns[i], add, positive/sample_num, positive, sample_num-(positive+negative)))
        print(f"                 :  1%={np.quantile(diff, 0.01):.4f}")
        #print(f"                 :  5%={np.quantile(diff, 0.05):.4f}")
        #print(f"                 : 95%={np.quantile(diff, 0.95):.4f}")
        print(f"                 : 99%={np.quantile(diff, 0.99):.4f}")
    print("")

    print("ETKF V.S. ETKFCC with est")
    for i in range(len(ns)):
        diff = df[f"opt_ETKFCC_with_est_n={ns[i]}_add={add:.1f}"].to_numpy() - df[f"opt_ETKF_n={ns[i]}_add={add:.1f}"].to_numpy()
        positive = np.count_nonzero(diff > 0)
        negative = np.count_nonzero(diff <= 0)
        print("n={}, add={:.1f} : p-value={}   (positive={}, check={})".format(ns[i], add, positive/sample_num, positive, sample_num-(positive+negative)))
        print(f"                 :  1%={np.quantile(diff, 0.01):.4f}")
        #print(f"                 :  5%={np.quantile(diff, 0.05):.4f}")
        #print(f"                 : 95%={np.quantile(diff, 0.95):.4f}")
        print(f"                 : 99%={np.quantile(diff, 0.99):.4f}")
    print("")

    print("ETKFCC with true V.S. ETKFCC with est")
    for i in range(len(ns)):
        diff = df[f"opt_ETKFCC_with_est_n={ns[i]}_add={add:.1f}"].to_numpy() - df[f"opt_ETKFCC_with_true_n={ns[i]}_add={add:.1f}"].to_numpy()
        positive = np.count_nonzero(diff > 0)
        negative = np.count_nonzero(diff <= 0)
        print("n={}, add={:.1f} : p-value={}   (positive={}, check={})".format(ns[i], add, positive/sample_num, positive, sample_num-(positive+negative)))
        print(f"                 :  1%={np.quantile(diff, 0.01):.4f}")
        #print(f"                 :  5%={np.quantile(diff, 0.05):.4f}")
        #print(f"                 : 95%={np.quantile(diff, 0.95):.4f}")
        print(f"                 : 99%={np.quantile(diff, 0.99):.4f}")
    print("")

if __name__ == "__main__":
    #check_estimated_parameters()
    #output_estimated_parameters(0.6, [1, 10, 30, 39])
    #show_optimal_ETKF_and_ETKFCC(0.6, [1, 10, 30, 39])

    #bootstrap_RMSE_ratio()
    p_test_RMSE_ratio()
    pass
