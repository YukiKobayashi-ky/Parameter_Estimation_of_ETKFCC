using LinearAlgebra
using Statistics
using Random
using Printf
using Base.Threads
using CSV
using DataFrames


"""
Numerical Integration
"""
const N = 40
const F = 8.0
function lorenz96!(dx_dt::Vector{Float64}, x::Vector{Float64}, (; F, N)::NamedTuple)
    F::Float64
    N::Int64

    for i=3:N-1
        dx_dt[i] = (x[i+1]-x[i-2])*x[i-1] - x[i] + F
    end
    dx_dt[1] = (x[2]-x[N-1])*x[N] - x[1] + F
    dx_dt[2] = (x[3]-x[N])*x[1] - x[2] + F
    dx_dt[N] = (x[1]-x[N-2])*x[N-1] - x[N] + F

    return nothing
end

function lorenz96!(dx_dt::Matrix{Float64}, x::Matrix{Float64}, (; F, N)::NamedTuple)
    F::Float64
    N::Int64

    for i=3:N-1
        dx_dt[i,:] = (x[i+1,:] .- x[i-2,:]) .* x[i-1,:] .- x[i,:] .+ F
    end

    dx_dt[1,:] = (x[2,:] .- x[N-1,:]) .* x[N,:] .- x[1,:] .+ F
    dx_dt[2,:] = (x[3,:] .- x[N,:]) .* x[1,:] .- x[2,:] .+ F
    dx_dt[N,:] = (x[1,:] .- x[N-2,:]) .* x[N-1,:] .- x[N,:] .+ F

    return nothing
end


struct RungeKutta4{F, T}
    dx_dt::F
    parameter::NamedTuple
    dt::Float64
    _k1::T
    _k2::T
    _k3::T
    _k4::T
end

function RungeKutta4(dx_dt::Function, parameter::NamedTuple, dt::Float64, n::Int64)
    _k1 = Vector{Float64}(undef, n)
    _k2 = Vector{Float64}(undef, n)
    _k3 = Vector{Float64}(undef, n)
    _k4 = Vector{Float64}(undef, n)
    return RungeKutta4(dx_dt, parameter, dt, _k1, _k2, _k3, _k4)
end

function RungeKutta4(dx_dt::Function, parameter::NamedTuple, dt::Float64, n::Int64, m::Int64)
    _k1 = Matrix{Float64}(undef, n, m)
    _k2 = Matrix{Float64}(undef, n, m)
    _k3 = Matrix{Float64}(undef, n, m)
    _k4 = Matrix{Float64}(undef, n, m)
    return RungeKutta4(dx_dt, parameter, dt, _k1, _k2, _k3, _k4)
end

function (rk4::RungeKutta4)(x_next::Vector{Float64}, x_pre::Vector{Float64})
    rk4.dx_dt(rk4._k1, x_pre, rk4.parameter)

    x_next .= x_pre .+ ((0.5 * rk4.dt) .* rk4._k1)
    rk4.dx_dt(rk4._k2, x_next, rk4.parameter)
    
    x_next .= x_pre .+ ((0.5 * rk4.dt) .* rk4._k2)
    rk4.dx_dt(rk4._k3, x_next, rk4.parameter)
    
    x_next .= x_pre .+ (rk4.dt .* rk4._k3)
    rk4.dx_dt(rk4._k4, x_next, rk4.parameter)
    
    @. x_next = x_pre + (rk4.dt / 6.0) * (rk4._k1 + (2.0 * rk4._k2) + (2.0 * rk4._k3) + rk4._k4)
    return nothing
end

function (rk4::RungeKutta4)(x_next::Matrix{Float64}, x_pre::Matrix{Float64})
    rk4.dx_dt(rk4._k1, x_pre, rk4.parameter)

    x_next .= x_pre .+ ((0.5 * rk4.dt) .* rk4._k1)
    rk4.dx_dt(rk4._k2, x_next, rk4.parameter)
    
    x_next .= x_pre .+ ((0.5 * rk4.dt) .* rk4._k2)
    rk4.dx_dt(rk4._k3, x_next, rk4.parameter)
    
    x_next .= x_pre .+ (rk4.dt .* rk4._k3)
    rk4.dx_dt(rk4._k4, x_next, rk4.parameter)
    
    @. x_next = x_pre + (rk4.dt / 6.0) * (rk4._k1 + (2.0 * rk4._k2) + (2.0 * rk4._k3) + rk4._k4)
    return nothing
end



"""
ETKF
"""
struct ETKF{T}
    ρ::Float64
    R_inv::T
    m::Int64
end

function (etkf::ETKF)(Xf, y)
    xf_mean = vec(sum(Xf, dims=2)) ./ Float64(etkf.m)
    dXf = Xf .- xf_mean
    dYf = dXf
    
    dob = y .- xf_mean
    Xa = similar(Xf)
    I_m = I(etkf.m)
    
    Pa_inv = ((etkf.m - 1)/etkf.ρ) * I_m + (dYf' * etkf.R_inv * dYf)
    
    D, U = eigen(Symmetric(Pa_inv)) # D, U = eigen(UDU')
    
    mean_update = vec(dXf *  U * ( (1.0 ./ D) .* U') * dYf' * etkf.R_inv * dob)
    dXa = dXf * U * ( (sqrt(etkf.m - 1) ./ sqrt.(D)) .* U' )
    
    xa_mean = xf_mean .+ mean_update
    Xa = xa_mean .+ dXa

    # --- 誤差共分散統計 (B, A) の計算 ---
    HBH = diag((dXf*dXf')./(etkf.m - 1))
    HAH = diag((dXa*dXa')./(etkf.m - 1))

    # --- イノベーション統計 ---
    doa = y .- xa_mean
    dab = mean_update
    
    ob_ob = dob .* dob
    oa_oa = doa .* doa
    ab_ab = dab .* dab
    ab_ob = dab .* dob
    oa_ob = doa .* dob
    ab_oa = dab .* doa

    stats = (
        HBH, HAH,
        ob_ob, oa_oa, ab_ab, 
        ab_ob, oa_ob, ab_oa,
    )
    
    return Xa, stats
end


function run_etkf_npoint(ρ, r, add, n, start_ensemble, eta, true_data)
    m = size(start_ensemble,2)
    r_inv = ones(Float64, N)
    r_inv[1:n] ./= r
    R_inv = Diagonal(r_inv)
    
    parameter = (F=F, N=N,)
    step_matrix! = RungeKutta4(lorenz96!, parameter, 0.05, N, m)
    etkf = ETKF(ρ, R_inv, m)

    Xf = copy(start_ensemble)
    Xf_next = Matrix{Float64}(undef, N, m)
    Xa = copy(Xf)

    k = 1
    for _ in 1:4*365
        step_matrix!(Xf_next, Xf)

        y = copy(true_data[:,k])
        xf_mean = vec(sum(Xf_next, dims=2)) ./ Float64(m)
        d = xf_mean .- true_data[:,k]
        y[1:n] .+= add * d[1:n]
        y .+= eta[:,k]

        Xa, _ = etkf(Xf_next, y)

        Xf = copy(Xa)
        k += 1
    end

    T = 4*365*10
    diff_all = Matrix{Float64}(undef, N*3, 0)
    diff_tmp = Vector{Float64}(undef, N*3)
    for t in 1:T
        step_matrix!(Xf_next, Xf)

        xf_mean = vec(sum(Xf_next, dims=2)) ./ Float64(m)
        d = xf_mean .- true_data[:,k]
        diff_tmp[1:N] .= d

        y = copy(true_data[:,k])
        y[1:n] .+= add * d[1:n]
        y .+= eta[:,k]

        d = y .- true_data[:,k]
        diff_tmp[N+1:2*N] .= d

        Xa, _ = etkf(Xf_next, y)
        xa_mean = vec(sum(Xa, dims=2)) ./ Float64(m)
        d = xa_mean .- true_data[:,k]
        diff_tmp[2*N+1:end] .= d
        
        diff_all = hcat(diff_all, copy(diff_tmp))

        Xf = copy(Xa)
        k += 1
    end

    name = vcat(
        [Symbol("error_xf_$(i)") for i in 1:N],
        [Symbol("error_y_$(i)") for i in 1:N],
        [Symbol("error_xa_$(i)") for i in 1:N],
    )
    df = DataFrame(transpose(diff_all), name)
    CSV.write("./point_estimation/"*@sprintf("optimal_timeseries_etkf_%dpoint", n)*@sprintf("_add=%.1f.csv", add),df)
end

function optimal_etkf_timeseries()
    add = 0.6

    tuning_parameters = CSV.read(@sprintf("./point_estimation/etkf_optimal_tuning_parameters_add=%.1f.csv", add), DataFrame)
    ks = tuning_parameters.k
    ρs = tuning_parameters.multi_inf
    rs = tuning_parameters.r

    eta = CSV.read("./eta.csv", DataFrame; header=false) |> Matrix |> transpose |> copy
    start_ensemble = CSV.read("./start_ensemble.csv", DataFrame; header=false) |> Matrix
    true_data = CSV.read("./true_data.csv", DataFrame; header=false) |> Matrix |> transpose |> copy

    Threads.@threads for i in  1:length(ks)
        run_etkf_npoint(ρs[i], rs[i], add, ks[i], start_ensemble, eta, true_data)
    end
    
end

#optimal_etkf_timeseries()



"""
ETKFCC
"""
struct ETKFCC{S,T}
    ρ::Float64
    A::S
    Ruc_inv::T
    m::Int64
end

function (etkfcc::ETKFCC)(Xf, y)
    xf_mean = vec(sum(Xf, dims=2)) ./ Float64(etkfcc.m)
    dXf = Xf .- xf_mean
    dYf = dXf
    
    dob = y .- xf_mean
    Xa = similar(Xf)
    I_m = I(etkfcc.m)
    IA = I(size(etkfcc.A,1)) - etkfcc.A
    
    Pa_inv = ((etkfcc.m - 1)/etkfcc.ρ) * I_m + (dYf' * IA' * etkfcc.Ruc_inv * IA * dYf)
    
    D, U = eigen(Symmetric(Pa_inv)) # D, U = eigen(UDU')
    
    mean_update = vec(dXf *  U * ( (1.0 ./ D) .* U') * dYf' * IA' * etkfcc.Ruc_inv * dob)
    dXa = dXf * U * ( (sqrt(etkfcc.m - 1) ./ sqrt.(D)) .* U' )
    
    xa_mean = xf_mean .+ mean_update
    Xa = xa_mean .+ dXa

    # --- 誤差共分散統計 (B, A) の計算 ---
    HBH = diag((dXf*dXf')./(etkfcc.m - 1))
    HAH = diag((dXa*dXa')./(etkfcc.m - 1))

    # --- イノベーション統計 ---
    doa = y .- xa_mean
    dab = mean_update
    
    ob_ob = dob .* dob
    oa_oa = doa .* doa
    ab_ab = dab .* dab
    ab_ob = dab .* dob
    oa_ob = doa .* dob
    ab_oa = dab .* doa

    stats = (
        HBH, HAH,
        ob_ob, oa_oa, ab_ab, 
        ab_ob, oa_ob, ab_oa,
    )
    
    return Xa, stats
end


function run_ETKFCC(ρ, ruc_sys, add_sys, add_true, start_ensemble, eta, true_data, name, n, add)
    m = size(start_ensemble,2)
    Ruc_inv = Diagonal(inv.(ruc_sys))
    A = Diagonal(add_sys)
    

    parameter = (F=F, N=N,)
    step_matrix! = RungeKutta4(lorenz96!, parameter, 0.05, N, m)
    etkfcc = ETKFCC(ρ, A, Ruc_inv, m)

    Xf = copy(start_ensemble)
    Xf_next = Matrix{Float64}(undef, N, m)
    Xa = copy(Xf)

    k = 1
    for _ in 1:4*365
        step_matrix!(Xf_next, Xf)

        y = copy(true_data[:,k])
        xf_mean = vec(sum(Xf_next, dims=2)) ./ Float64(m)
        d = xf_mean .- true_data[:,k]
        y .+= add_true .* d
        y .+= eta[:,k]

        Xa, _ = etkfcc(Xf_next, y)

        Xf = copy(Xa)
        k += 1
    end

    T = 4*365*10
    diff_all = Matrix{Float64}(undef, N*3, 0)
    diff_tmp = Vector{Float64}(undef, N*3)
    for t in 1:T
        step_matrix!(Xf_next, Xf)

        xf_mean = vec(sum(Xf_next, dims=2)) ./ Float64(m)
        d = xf_mean .- true_data[:,k]
        diff_tmp[1:N] .= d

        y = copy(true_data[:,k])
        y .+= add_true .* d
        y .+= eta[:,k]

        d = y .- true_data[:,k]
        diff_tmp[N+1:2*N] .= d

        Xa, _ = etkfcc(Xf_next, y)
        xa_mean = vec(sum(Xa, dims=2)) ./ Float64(m)
        d = xa_mean .- true_data[:,k]
        diff_tmp[2*N+1:end] .= d

        diff_all = hcat(diff_all, copy(diff_tmp))

        Xf = copy(Xa)
        k += 1
    end

    column = vcat(
        [Symbol("error_xf_$(i)") for i in 1:N],
        [Symbol("error_y_$(i)") for i in 1:N],
        [Symbol("error_xa_$(i)") for i in 1:N],
    )
    df = DataFrame(transpose(diff_all), column)
    CSV.write("./point_estimation_cc/optimal_timeseries_etkfcc_"*name*@sprintf("_%dpoint", n)*@sprintf("_add=%.1f.csv", add),df)

end

function optimal_etkfcc_with_true_timeseries()
    add = 0.6
    tuning_parameters = CSV.read(@sprintf("./point_estimation_cc/etkfcc_with_true_optimal_tuning_parameters_add=%.1f.csv", add), DataFrame)
    ks = tuning_parameters.k
    ρs = tuning_parameters.multi_inf

    eta = CSV.read("./eta.csv", DataFrame; header=false) |> Matrix |> transpose |> copy
    start_ensemble = CSV.read("./start_ensemble.csv", DataFrame; header=false) |> Matrix
    true_data = CSV.read("./true_data.csv", DataFrame; header=false) |> Matrix |> transpose |> copy

    Threads.@threads for i in  1:length(ks)
        add_true = zeros(Float64, N)
        add_true[1:ks[i]] .= add
        add_sys = zeros(Float64, N)
        add_sys[1:ks[i]] .= add
        ruc_sys = ones(Float64, N)
        run_ETKFCC(ρs[i], ruc_sys, add_sys, add_true, start_ensemble, eta, true_data, "with_true", ks[i], add)
    end
end

#optimal_etkfcc_with_true_timeseries()

function optimal_etkfcc_with_est_timeseries()
    add = 0.6
    tuning_parameters = CSV.read(@sprintf("./point_estimation_cc/etkfcc_with_est_optimal_tuning_parameters_add=%.1f.csv", add), DataFrame)
    ks = tuning_parameters.k
    ρs = tuning_parameters.multi_inf

    eta = CSV.read("./eta.csv", DataFrame; header=false) |> Matrix |> transpose |> copy
    start_ensemble = CSV.read("./start_ensemble.csv", DataFrame; header=false) |> Matrix
    true_data = CSV.read("./true_data.csv", DataFrame; header=false) |> Matrix |> transpose |> copy

    Threads.@threads for i in  1:length(ks)
        add_true = zeros(Float64, N)
        add_true[1:ks[i]] .= add
        parameters = CSV.read("./point_estimation/"*@sprintf("estimated_parameters_%dpoint", ks[i])*@sprintf("_add=%.1f.csv", add), DataFrame)
        add_sys = parameters.add_est
        ruc_sys = parameters.ruc_est
        run_ETKFCC(ρs[i], ruc_sys, add_sys, add_true, start_ensemble, eta, true_data, "with_est", ks[i], add)
    end
end

#optimal_etkfcc_with_est_timeseries()
