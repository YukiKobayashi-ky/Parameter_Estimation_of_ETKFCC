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

plt.rcParams['mathtext.fontset'] = 'cm'

def get_optimal(df):
    df_calc = df.query('rmse_a > 0.0')
    df_calc["RMSEratio"] = df_calc["rmse_a"] / df_calc["rmse_o"]

    return df_calc.loc[df_calc["rmse_a"].idxmin()]

def output_estimated_parameters(add, num):
    for i in range(len(num)):
        df = pd.read_csv(f"./point_estimation/etkf_{num[i]}point_add=0.6.csv", comment='#')
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

def show_all_figure_point_est(add, num_est, num_rmseratio):
    fig = plt.figure(figsize = (10, 24))
        
    ax1 = fig.add_subplot(3,1,1)
    ax1.set_title("(a) Point estimation : $a^{est}_i$", fontsize=20)
    ax1.set_xlabel("Location index $i$", fontsize=20)
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
    ax1.hlines(add, xmin=0,xmax=41, colors='black', linewidth=0.8)
    ax1.vlines(10, ymin=-1,ymax=2, colors="black", linewidth=0.8)
    ax1.vlines(20, ymin=-1,ymax=2, colors="black", linewidth=0.8)
    ax1.vlines(30, ymin=-1,ymax=2, colors="black", linewidth=0.8)

    ax2 = fig.add_subplot(3,1,2)
    ax2.set_title("(b) Point estimation : $r^{uc\_est}_i$", fontsize=20)
    ax2.set_xlabel("Location index $i$", fontsize=20)
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
    ax2.vlines(20, ymin=-1,ymax=2, colors="black", linewidth=0.8)
    ax2.vlines(30, ymin=-1,ymax=2, colors="black", linewidth=0.8)

    for k in num_est:
        df = pd.read_csv(f"./point_estimation/etkf_{k}point_add=0.6.csv", comment='#')
        df_optimal = get_optimal(df)
    
        HBH = df_optimal.filter(like="HBH").to_numpy()
        inf = df_optimal["multi_inf"]
        HBH = inf * HBH
        ob_ob = df_optimal.filter(like="ob_ob").to_numpy()
        ab_ob = df_optimal.filter(like="ab_ob").to_numpy()
    
        a_est = 1.0 - ab_ob/HBH
        ruc_est = ob_ob - ab_ob*ab_ob/HBH
    
        ax1.plot(np.arange(1,41), a_est, "-o", label=f"$k={k}$", zorder=40-k)
        ax2.plot(np.arange(1,41), ruc_est, "-o", label=f"$k={k}$", zorder=40-k)

    ax1.legend(loc="upper right", fontsize=18)
    ax2.legend(loc="upper right", fontsize=18)


    ax3 = fig.add_subplot(3,1,3)
    ax3.set_title("(c) RMSE ratio with point estimation", fontsize=20)
    ax3.set_xlabel(f"Number of $a^{{true}}_i={add}$ location $k$", fontsize=20)
    x_major_tick = [0, 10, 20, 30, 40]
    x_minor_tick = np.arange(1,40)
    ax3.set_xticks(x_major_tick,minor=False)
    ax3.set_xticklabels(["{}".format(x_major_tick[i]) for i in range(len(x_major_tick))],fontsize=20)
    ax3.set_xticks(x_minor_tick,minor=True)
    ax3.set_xlim(0.0,40.0)
    ax3.set_ylabel("RMSE ratio", fontsize=20)
    y_major_tick = [0, 0.2, 0.4, 0.6, 0.8, 1]
    y_minor_tick = np.arange(0.0,1.0,0.1)
    ax3.set_yticks(y_major_tick,minor=False)
    ax3.set_yticklabels(["{}".format(y_major_tick[i]) for i in range(len(y_major_tick))],fontsize=20)
    ax3.set_yticks(y_minor_tick,minor=True)
    ax3.set_ylim(0.0,1.0)
    ax3.vlines(10, ymin=-1,ymax=2, colors="black", linewidth=0.8)
    ax3.vlines(20, ymin=-1,ymax=2, colors="black", linewidth=0.8)
    ax3.vlines(30, ymin=-1,ymax=2, colors="black", linewidth=0.8)

    rmse_ratio = []
    for k in num_rmseratio:
        df = pd.read_csv(f"./point_estimation/etkf_{k}point_add={add:.1f}.csv", comment='#')
        df_optimal = get_optimal(df)
        rmse_ratio.append(df_optimal["RMSEratio"])
    ax3.plot(num_rmseratio, rmse_ratio, "-^", color="gray", label="ETKF", zorder=0)

    rmse_ratio = []
    for k in num_rmseratio:
        df = pd.read_csv(f"./point_estimation_cc/etkfcc_with_true_{k}point_add={add:.1f}.csv", comment='#')
        df_optimal = get_optimal(df)
        rmse_ratio.append(df_optimal["RMSEratio"])
    ax3.plot(num_rmseratio, rmse_ratio, "-v", color="black", label="ETKFCC (true parameter)", zorder=1)

    df_bootstrap = pd.read_csv('./RMSE_ratio_sample_point_est.csv', comment='#')
    rmse_ratio = []
    for k in num_rmseratio:
        df = pd.read_csv(f"./point_estimation_cc/etkfcc_with_est_{k}point_add={add:.1f}.csv", comment='#')
        df_optimal = get_optimal(df)
        rmse_ratio_cc_est = df_optimal["RMSEratio"]
        rmse_ratio.append(rmse_ratio_cc_est)
        df = pd.read_csv(f"./point_estimation/etkf_{k}point_add={add:.1f}.csv", comment='#')
        df_optimal = get_optimal(df)
        rmse_ratio_etkf = df_optimal["RMSEratio"]
        diff = df_bootstrap[f"opt_ETKFCC_with_est_k={k}_add={add:.1f}"].to_numpy() - df_bootstrap[f"opt_ETKF_k={k}_add={add:.1f}"].to_numpy()
        if (np.count_nonzero(diff < 0) / df.shape[0] > 0.99) and (1 - rmse_ratio_cc_est/rmse_ratio_etkf > 0.05):
            facecolor = "white"
        else:
            facecolor = "red"
        ax3.scatter(k, rmse_ratio_cc_est, s=40, c=facecolor, edgecolors='red', zorder=3)
    ax3.plot(num_rmseratio, rmse_ratio, "-o", color="red", label="ETKFCC (estimated parameter)", zorder=2)
    
    ax3.legend(loc="upper right", fontsize=18)

    fig.subplots_adjust(bottom=0.04,top=0.98,left=0.12,right=0.98)
    plt.savefig("point_est_figure.png")
    #plt.show()

def output_optimal_tuning(add, num):
    col_add = []
    col_k = []
    col_inf = []
    col_r = []
    for k in num:
        df = pd.read_csv(f"./point_estimation/etkf_{k}point_add={add:.1f}.csv", comment='#')
        df_optimal = get_optimal(df)
        col_add.append(add)
        col_k.append(k)
        col_inf.append(df_optimal["multi_inf"])
        col_r.append(df_optimal["r"])
    df = pd.DataFrame({
        'add':col_add,
        'k':col_k,
        'multi_inf':col_inf,
        'r':col_r,
    })
    df.to_csv(f"./point_estimation/etkf_optimal_tuning_parameters_add={add:.1f}.csv", index=False)

    col_add = []
    col_k = []
    col_inf = []
    col_r = []
    for k in num:
        df = pd.read_csv(f"./point_estimation_cc/etkfcc_with_true_{k}point_add={add:.1f}.csv", comment='#')
        df_optimal = get_optimal(df)
        col_add.append(add)
        col_k.append(k)
        col_inf.append(df_optimal["multi_inf"])
    df = pd.DataFrame({
        'add':col_add,
        'k':col_k,
        'multi_inf':col_inf,
    })
    df.to_csv(f"./point_estimation_cc/etkfcc_with_true_optimal_tuning_parameters_add={add:.1f}.csv", index=False)

    col_add = []
    col_k = []
    col_inf = []
    col_r = []
    for k in num:
        df = pd.read_csv(f"./point_estimation_cc/etkfcc_with_est_{k}point_add={add:.1f}.csv", comment='#')
        df_optimal = get_optimal(df)
        col_add.append(add)
        col_k.append(k)
        col_inf.append(df_optimal["multi_inf"])
    df = pd.DataFrame({
        'add':col_add,
        'k':col_k,
        'multi_inf':col_inf,
    })
    df.to_csv(f"./point_estimation_cc/etkfcc_with_est_optimal_tuning_parameters_add={add:.1f}.csv", index=False)



### p test ###
def bootstrap_RMSE_ratio_unit(df):
    df_sample = df.sample(n=df.shape[0], replace=True)
    rmse_xa = np.sqrt(np.mean(df_sample.filter(like="xa").to_numpy(), axis=0))
    rmse_y = np.sqrt(np.mean(df_sample.filter(like="y").to_numpy(), axis=0))
    rmse_ratio = rmse_xa / rmse_y
    #print(rmse_ratio)
    return rmse_ratio

def bootstrap_RMSE_ratio(add, num):
    print("######  sampling RMSE-ratio with bootstrap (point estimation)  ######")
    sample_num = int(1e+05)

    data = np.empty([sample_num,0])
    line = ""

    for k in num:
        print(f"add={add:.1f} : k={k}")
        optimal_etkf_time_series = pd.read_csv(f"./point_estimation/optimal_timeseries_etkf_{k}point_add={add:.1f}.csv", comment='#')
        etkf_mse_xa = np.mean(optimal_etkf_time_series.filter(like="error_xa").to_numpy()**2, axis=1)
        etkf_mse_y = np.mean(optimal_etkf_time_series.filter(like="error_y").to_numpy()**2, axis=1)
        optimal_etkfcc_with_true_time_series = pd.read_csv(f"./point_estimation_cc/optimal_timeseries_etkfcc_with_true_{k}point_add={add:.1f}.csv", comment='#')
        etkfcc_with_true_mse_xa = np.mean(optimal_etkfcc_with_true_time_series.filter(like="error_xa").to_numpy()**2, axis=1)
        etkfcc_with_true_mse_y = np.mean(optimal_etkfcc_with_true_time_series.filter(like="error_y").to_numpy()**2, axis=1)
        optimal_etkfcc_with_est_time_series = pd.read_csv(f"./point_estimation_cc/optimal_timeseries_etkfcc_with_est_{k}point_add={add:.1f}.csv", comment='#')
        etkfcc_with_est_mse_xa = np.mean(optimal_etkfcc_with_est_time_series.filter(like="error_xa").to_numpy()**2, axis=1)
        etkfcc_with_est_mse_y = np.mean(optimal_etkfcc_with_est_time_series.filter(like="error_y").to_numpy()**2, axis=1)

        time_series = pd.DataFrame({
            'etkf_mse_xa':etkf_mse_xa,
            'etkf_mse_y':etkf_mse_y,
            'etkfcc_with_true_mse_xa':etkfcc_with_true_mse_xa,
            'etkfcc_with_true_mse_y':etkfcc_with_true_mse_y,
            'etkfcc_with_est_mse_xa':etkfcc_with_est_mse_xa,
            'etkfcc_with_est_mse_y':etkfcc_with_est_mse_y,
        })
        with multiprocessing.Pool(processes=16) as pool:
            result = pool.map(bootstrap_RMSE_ratio_unit,[time_series for i in range(sample_num)])
        data = np.hstack([data,result])
        line = line + f"opt_ETKF_k={k}_add={add:.1f},opt_ETKFCC_with_true_k={k}_add={add:.1f},opt_ETKFCC_with_est_k={k}_add={add:.1f},"
    
    line = line.strip(",") + "\n"

    file = open("./RMSE_ratio_sample_point_est.csv", 'w')
    file.writelines("#sample_size={}\n".format(sample_num))
    file.writelines(line)
    np.savetxt(file, data, delimiter=',')
    file.close()

def p_test_RMSE_ratio(add, num):
    print("########### RMSE ratio #################")
    df = pd.read_csv('./RMSE_ratio_sample_point_est.csv', comment='#')
    sample_num = df.shape[0]
    add = 0.6

    print("ETKF V.S. ETKFCC with true")
    
    for k in num:
        diff =  df[f"opt_ETKFCC_with_true_k={k}_add={add:.1f}"].to_numpy() - df[f"opt_ETKF_k={k}_add={add:.1f}"].to_numpy()
        positive = np.count_nonzero(diff > 0)
        negative = np.count_nonzero(diff <= 0)
        print("n={}, add={:.1f} : p-value={}   (positive={}, check={})".format(k, add, positive/sample_num, positive, sample_num-(positive+negative)))
        print(f"                 :  1%={np.quantile(diff, 0.01):.4f}")
        #print(f"                 :  5%={np.quantile(diff, 0.05):.4f}")
        #print(f"                 : 95%={np.quantile(diff, 0.95):.4f}")
        print(f"                 : 99%={np.quantile(diff, 0.99):.4f}")
    print("")

    print("ETKF V.S. ETKFCC with est")
    for k in num:
        diff = df[f"opt_ETKFCC_with_est_k={k}_add={add:.1f}"].to_numpy() - df[f"opt_ETKF_k={k}_add={add:.1f}"].to_numpy()
        positive = np.count_nonzero(diff > 0)
        negative = np.count_nonzero(diff <= 0)
        print("n={}, add={:.1f} : p-value={}   (positive={}, check={})".format(k, add, positive/sample_num, positive, sample_num-(positive+negative)))
        print(f"                 :  1%={np.quantile(diff, 0.01):.4f}")
        #print(f"                 :  5%={np.quantile(diff, 0.05):.4f}")
        #print(f"                 : 95%={np.quantile(diff, 0.95):.4f}")
        print(f"                 : 99%={np.quantile(diff, 0.99):.4f}")
    print("")

    print("ETKFCC with true V.S. ETKFCC with est")
    for k in num:
        diff = df[f"opt_ETKFCC_with_est_k={k}_add={add:.1f}"].to_numpy() - df[f"opt_ETKFCC_with_true_k={k}_add={add:.1f}"].to_numpy()
        positive = np.count_nonzero(diff > 0)
        negative = np.count_nonzero(diff <= 0)
        print("n={}, add={:.1f} : p-value={}   (positive={}, check={})".format(k, add, positive/sample_num, positive, sample_num-(positive+negative)))
        print(f"                 :  1%={np.quantile(diff, 0.01):.4f}")
        #print(f"                 :  5%={np.quantile(diff, 0.05):.4f}")
        #print(f"                 : 95%={np.quantile(diff, 0.95):.4f}")
        print(f"                 : 99%={np.quantile(diff, 0.99):.4f}")
    print("")



def show_est_figure_point_est(add, num_est):
    fig = plt.figure(figsize = (20, 12))
        
    ax1 = fig.add_subplot(1,2,1)
    ax1.set_title("(a) Point estimation : $a^{est}_i$", fontsize=20)
    ax1.set_xlabel("Location index $i$", fontsize=20)
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
    ax1.hlines(add, xmin=0,xmax=41, colors='black', linewidth=0.8)
    ax1.vlines(10, ymin=-1,ymax=2, colors="black", linewidth=0.8)
    ax1.vlines(20, ymin=-1,ymax=2, colors="black", linewidth=0.8)
    ax1.vlines(30, ymin=-1,ymax=2, colors="black", linewidth=0.8)

    ax2 = fig.add_subplot(1,2,2)
    ax2.set_title("(b) Point estimation : $r^{uc\_est}_i$", fontsize=20)
    ax2.set_xlabel("Location index $i$", fontsize=20)
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
    ax2.vlines(20, ymin=-1,ymax=2, colors="black", linewidth=0.8)
    ax2.vlines(30, ymin=-1,ymax=2, colors="black", linewidth=0.8)

    for k in num_est:
        df = pd.read_csv(f"./point_estimation/etkf_{k}point_add=0.6.csv", comment='#')
        df_optimal = get_optimal(df)
    
        HBH = df_optimal.filter(like="HBH").to_numpy()
        inf = df_optimal["multi_inf"]
        HBH = inf * HBH
        ob_ob = df_optimal.filter(like="ob_ob").to_numpy()
        ab_ob = df_optimal.filter(like="ab_ob").to_numpy()
    
        a_est = 1.0 - ab_ob/HBH
        ruc_est = ob_ob - ab_ob*ab_ob/HBH
    
        ax1.plot(np.arange(1,41), a_est, "-o", label=f"$k={k}$", zorder=40-k)
        ax2.plot(np.arange(1,41), ruc_est, "-o", label=f"$k={k}$", zorder=40-k)

    ax1.legend(loc="upper right", fontsize=18)
    ax2.legend(loc="upper right", fontsize=18)

    fig.subplots_adjust(bottom=0.1,top=0.9,left=0.1,right=0.9)
    plt.savefig("point_est_each.png")
    #plt.show()

if __name__ == "__main__":
    #output_estimated_parameters(0.6, [k for k in range(41)])
    show_all_figure_point_est(0.6, [1, 10, 30, 39], [k for k in range(41)])
    #output_optimal_tuning(0.6, [k for k in range(41)])

    #bootstrap_RMSE_ratio(0.6, [k for k in range(41)])
    #p_test_RMSE_ratio(0.6, [k for k in range(41)])

    #show_est_figure_point_est(0.6, [30+i for i in range(11)])

    pass
