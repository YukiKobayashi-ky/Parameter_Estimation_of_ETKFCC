from concurrent.futures import process
import numpy as np
import numpy.linalg as LA
import multiprocessing
import time
import warnings
import os

warnings.simplefilter('error')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

###############################################################
# Lorenz-96
###############################################################
N = 40   # 変数の個数
F = 8.0  # パラメータ
DT = 0.05 # 時間刻み幅

def dX_dt(ary):
    xj_zero = ary.copy()
    xj_one = np.roll(ary, -1)
    xj_m_two = np.roll(ary, 2)
    xj_m_one = np.roll(ary, 1)
    dX_dt = (xj_one - xj_m_two)*xj_m_one - xj_zero + np.ones(N)*F
    return dX_dt

def dX_dt_matrix(ary, m):
    xj_zero = ary.copy()
    xj_one = np.roll(ary, -1, axis=0)
    xj_m_two = np.roll(ary, 2, axis=0)
    xj_m_one = np.roll(ary, 1, axis=0)
    dX_dt = (xj_one - xj_m_two)*xj_m_one - xj_zero + np.ones((N,m))*F
    return dX_dt

def step(x):
    for i in range(5):
        k1 = 0.01 * dX_dt(x)
        k2 = 0.01 * dX_dt(x + k1/2)
        k3 = 0.01 * dX_dt(x + k2/2)
        k4 = 0.01 * dX_dt(x + k3)
        x = x + (k1 + 2*k2 + 2*k3 + k4)/6
    return x

def step_matrix(X, m):
    k1 = DT * dX_dt_matrix(X, m)
    k2 = DT * dX_dt_matrix(X + k1/2, m)
    k3 = DT * dX_dt_matrix(X + k2/2, m)
    k4 = DT * dX_dt_matrix(X + k3, m)
    X = X + (k1 + 2*k2 + 2*k3 + k4)/6
    return X



#############################################################
# フィルタ
#############################################################
### 本体 ###
def ETKF(X_forcast, m, H, y, R, inf, add):
    # 偏差の生成
    xm_forcast  = (np.mean(X_forcast, axis = 1)).reshape(N,1)
    dX_f = ( X_forcast - np.tile(xm_forcast, (1,m)) )
    dY_d = np.dot(H,dX_f)

    # 各変数の解析
    R_inv = np.diag(np.reciprocal(R))
    UDU = ((m-1)/inf)*np.eye(m) + (np.dot(dY_d.T, np.dot(R_inv,dY_d))) # Multiplicative inflation
    D, U = LA.eigh(UDU)
    D_inv = np.reciprocal(D)
    sqrtD_inv = np.diag(np.sqrt(D_inv))
    T = np.sqrt(m-1)*(np.dot(U,np.dot(sqrtD_inv,U.T)))
    dX_a = dX_f @ T
    update = np.dot(dX_f,(np.dot((np.dot(T,T.T)/(m-1)),np.dot(dY_d.T,np.dot(R_inv,np.tile(y.reshape(H.shape[0],1) - np.dot(H,xm_forcast), (1,m))))) + T))
    X_analise = np.tile(xm_forcast, (1,m)) + update

    B_mat = (dX_f @ dX_f.T) / (m-1)
    B = np.trace(B_mat)/N
    A_mat = (dX_a @ dX_a.T) / (m-1)
    A = np.trace(A_mat)/N
    ## Innovation Statistics ##
    # 基本の差分
    d_ob = (y.reshape(40,1) - np.dot(H,xm_forcast)).reshape(40)
    d_oa = (y.reshape(40,1) - np.dot(H,np.mean(X_analise, axis = 1).reshape(N,1))).reshape(40)
    d_ab = (np.dot(H,np.mean(X_analise, axis = 1).reshape(40,1)) - np.dot(H,xm_forcast)).reshape(40)

    # 各種統計量
    ob_ob = np.multiply(d_ob,d_ob)
    oa_oa = np.multiply(d_oa,d_oa)
    ab_ab = np.multiply(d_ab,d_ab)
    ab_ob = np.multiply(d_ab,d_ob)
    oa_ob = np.multiply(d_oa,d_ob)
    ab_oa = np.multiply(d_ab,d_oa)
    HBH = np.diag(H @ B_mat @H.T)
    HAH = np.diag(H @ A_mat @H.T)

    return X_analise,B,A, ob_ob,oa_oa,ab_ab,ab_ob,oa_ob,ab_oa,HBH,HAH

############################################################
# 観測の定義
############################################################
def make_all(num=0):
    H = np.eye(N)
    R = np.ones(N)
    return H, R

def make_add_error(X_forcast, eta, x_true, add, k, H):
    """
    X_forcast : アンサンブル
    m : アンサンブル数
    x_true : 真値
    add : addパラメータ
    i : 時刻(乱数参照用)
    """
    # 誤差の生成
    error_f_orig = np.mean(X_forcast, axis=1) - x_true
    error_o_add = (H@(add*error_f_orig)).reshape(H.shape[0]) + (H@eta[k,:].T).reshape(H.shape[0])
    return error_o_add

###############################################################
# 実行部分　
###############################################################
def run(list):
    x_true, X_start, m, inf, add, r, eta = list
    # 初期値
    X_analise = X_start
    H, R = make_all()
    R = r*R
    k = 0
    file = open("./short_estimation/time_series_add={:.1f}_inf={:.2f}_R={:.1f}.csv".format(add,inf,r), 'w')
    column_name = "time,forecast_RMSE,analysis_RMSE,observation_RMSE,correlation" \
                    + "," + ",".join([f"ob_ob_{i+1}" for i in range(40)]) \
                    + "," + ",".join([f"oa_oa_{i+1}" for i in range(40)]) \
                    + "," + ",".join([f"ab_ab_{i+1}" for i in range(40)]) \
                    + "," + ",".join([f"ab_ob_{i+1}" for i in range(40)]) \
                    + "," + ",".join([f"oa_ob_{i+1}" for i in range(40)]) \
                    + "," + ",".join([f"ab_oa_{i+1}" for i in range(40)]) \
                    + "," + ",".join([f"HBH_{i+1}" for i in range(40)]) \
                    + "," + ",".join([f"HAH_{i+1}" for i in range(40)]) \
                    + "\n"
    file.writelines(column_name)
    #file.close()
    try:
        # Spin up
        for i in range(1460):
            # 予測
            X_forcast = step_matrix(X_analise, m)
            error = make_add_error(X_forcast, eta, x_true[k,:], add, k, H) # 相関のある誤差
            y = H@(x_true[k,:].T) + error

            # 更新
            X_analise,B_tmp,A_tmp, ob_ob_tmp,oa_oa_tmp,ab_ab_tmp,ab_ob_tmp,oa_ob_tmp,ab_oa_tmp,HBH_tmp,HAH_tmp \
                = ETKF(X_forcast, m, H, y, R, inf, add)

            k += 1


        # 実行部分
        for i in range(1460*10):
            # 予測
            X_forcast = step_matrix(X_analise, m)
            f_error = LA.norm(np.mean(X_forcast, axis=1) - x_true[k,:]) / np.sqrt(N)
            error = make_add_error(X_forcast, eta, x_true[k,:], add, k, H) # 相関のある誤差
            o_error = LA.norm(error) / np.sqrt(N)
            cov = np.dot(error,np.mean(X_forcast, axis=1) - x_true[k,:]) / N
            correlation = cov / (o_error*f_error)
            y = H@(x_true[k,:].T)+ error

            # 更新
            X_analise,B_tmp,A_tmp, ob_ob_tmp,oa_oa_tmp,ab_ab_tmp,ab_ob_tmp,oa_ob_tmp,ab_oa_tmp,HBH_tmp,HAH_tmp \
                = ETKF(X_forcast, m, H, y, R, inf, add)

            # 出力
            a_error = LA.norm(np.mean(X_analise, axis = 1)-x_true[k,:]) / np.sqrt(N)
            np.savetxt(file, [np.hstack([[k,f_error,a_error,o_error,correlation],ob_ob_tmp,oa_oa_tmp,ab_ab_tmp,ab_ob_tmp,oa_ob_tmp,ab_oa_tmp,HBH_tmp,HAH_tmp])], delimiter=',')
            k += 1
    #"""
    except:
        print("Error add={:.1f} R={:.1f} inf={:.2f}".format(add,r,inf))
    else:
        print("Clear :: add_true={:.1f} R={:.1f} inf={:.2f}".format(add,r,inf))
    #"""
    file.close()
    return add, inf, r

###################################################################################
# 出力部分
###################################################################################
### 出力 ###
def output_all(m):
    x_true = np.loadtxt("./true_data.csv", delimiter=',')
    X_start = np.loadtxt("./start_ensemble.csv", delimiter=',')
    eta = np.loadtxt("./eta.csv", delimiter=',')

    start = time.time()

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

    list = []
    for i in range(len(parameters)):
        tmp_list = [x_true, X_start, m, parameters[i][1], parameters[i][0], parameters[i][2], eta]
        list.append(tmp_list)

    os.makedirs("./short_estimation", exist_ok=True)
    with multiprocessing.Pool(processes=10) as pool:
        result = pool.map(run,list)
    stop = time.time()
    print(stop-start)
    print("{:.1f}h".format((stop-start)/3600.0))

    ## 出力　##
    print("End Program")

### 本体 ###
if __name__ == "__main__":
    m = 40 # 走らせるアンサンブルの数

    output_all(m)
