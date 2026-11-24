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


"""
main
"""

function run_ETKFCC(ρ, ruc_sys, add_sys, add_true, start_ensemble, eta, true_data)
    m = size(start_ensemble,2)
    Ruc_inv = Diagonal(inv.(ruc_sys))
    A = Diagonal(add_sys)
    
    try
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
        stat_sum = zeros(Float64, N, 8)
        mse_y = Vector{Float64}(undef, T)
        mse_xa = Vector{Float64}(undef, T)
        mse_xf = Vector{Float64}(undef, T)
        for t in 1:T
            step_matrix!(Xf_next, Xf)

            xf_mean = vec(sum(Xf_next, dims=2)) ./ Float64(m)
            d = xf_mean .- true_data[:,k]
            mse_xf[t] = sum(d.*d) / Float64(N)

            y = copy(true_data[:,k])
            y .+= add_true .* d
            y .+= eta[:,k]

            d = y .- true_data[:,k]
            mse_y[t] = sum(d.*d) / Float64(N)

            Xa, stat = etkfcc(Xf_next, y)
            xa_mean = vec(sum(Xa, dims=2)) ./ Float64(m)
            d = xa_mean .- true_data[:,k]
            mse_xa[t] = sum(d.*d) / Float64(N)
            
            for i in 1:8
                stat_sum[:, i] .+= stat[i]
            end

            Xf = copy(Xa)
            k += 1
        end

        stat_sum .= stat_sum ./ Float64(T)

        result = Vector{Float64}(undef, 4 + N*3 + N*8)
        result[1] = ρ
        result[2] = sqrt(mean(mse_xa))
        result[3] = sqrt(mean(mse_xf))
        result[4] = sqrt(mean(mse_y))
        result[5:N+4] = add_true
        result[N+5:2*N+4] = add_sys
        result[2*N+5:3*N+4] = ruc_sys
        result[3*N+5:end] .= vec(stat_sum)
        
        return result
    catch
        result = Vector{Float64}(undef, 4 + N*3 + N*8)
        result[1] = ρ
        result[2] = 0.0
        result[3] = 0.0
        result[4] = 0.0
        result[5:N+4] = add_true
        result[N+5:2*N+4] = add_sys
        result[2*N+5:3*N+4] = ruc_sys
        result[3*N+5:end] .= zeros(Float64, N*8)
        return result
    end
end


function output_npoint_ETKFCC_with_true(add, n)
    println("### start spatial ETKFCC with true : add=$(add), n=$(n)")
    ρs = [1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10]
    add_true = zeros(Float64, N)
    add_true[1:n] .= add
    add_sys = zeros(Float64, N)
    add_sys[1:n] .= add
    ruc_sys = ones(Float64, N)

    eta = CSV.read("./eta.csv", DataFrame; header=false) |> Matrix |> transpose |> copy
    start_ensemble = CSV.read("./start_ensemble.csv", DataFrame; header=false) |> Matrix
    true_data = CSV.read("./true_data.csv", DataFrame; header=false) |> Matrix |> transpose |> copy

    results = zeros(Float64, length(ρs), 4 + N*3 + N*8)

    Threads.@threads for i in  1:length(ρs)
        results[i,:] .= run_ETKFCC(ρs[i], ruc_sys, add_sys, add_true, start_ensemble, eta, true_data)
    end

    name = [:multi_inf, :rmse_a, :rmse_f, :rmse_o]
    append!(name, [Symbol("add_true_$(i)") for i in 1:N])
    append!(name, [Symbol("add_sys_$(i)") for i in 1:N])
    append!(name, [Symbol("ruc_sys_$(i)") for i in 1:N])
    append!(name, [Symbol("HBH_$(i)") for i in 1:N])
    append!(name, [Symbol("HAH_$(i)") for i in 1:N])
    append!(name, [Symbol("ob_ob_$(i)") for i in 1:N])
    append!(name, [Symbol("oa_oa_$(i)") for i in 1:N])
    append!(name, [Symbol("ab_ab_$(i)") for i in 1:N])
    append!(name, [Symbol("ab_ob_$(i)") for i in 1:N])
    append!(name, [Symbol("oa_ob_$(i)") for i in 1:N])
    append!(name, [Symbol("ab_oa_$(i)") for i in 1:N])

    df = DataFrame(results, name)
    CSV.write("./point_estimation_cc/"*@sprintf("etkfcc_with_true_%dpoint", n)*@sprintf("_add=%.1f.csv", add), df)
end

function output_npoint_ETKFCC_with_est(add, n)
    println("### start spatial ETKFCC with est : add=$(add), n=$(n)")
    ρs = [1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10]
    add_true = zeros(Float64, N)
    add_true[1:n] .= add

    parameters = CSV.read("./point_estimation/"*@sprintf("estimated_parameters_%dpoint", n)*@sprintf("_add=%.1f.csv", add), DataFrame)
    add_sys = parameters.add_est
    ruc_sys = parameters.ruc_est
    
    eta = CSV.read("./eta.csv", DataFrame; header=false) |> Matrix |> transpose |> copy
    start_ensemble = CSV.read("./start_ensemble.csv", DataFrame; header=false) |> Matrix
    true_data = CSV.read("./true_data.csv", DataFrame; header=false) |> Matrix |> transpose |> copy

    results = zeros(Float64, length(ρs), 4 + N*3 + N*8)

    Threads.@threads for i in  1:length(ρs)
        results[i,:] .= run_ETKFCC(ρs[i], ruc_sys, add_sys, add_true, start_ensemble, eta, true_data)
    end

    name = [:multi_inf, :rmse_a, :rmse_f, :rmse_o]
    append!(name, [Symbol("add_true_$(i)") for i in 1:N])
    append!(name, [Symbol("add_sys_$(i)") for i in 1:N])
    append!(name, [Symbol("ruc_sys_$(i)") for i in 1:N])
    append!(name, [Symbol("HBH_$(i)") for i in 1:N])
    append!(name, [Symbol("HAH_$(i)") for i in 1:N])
    append!(name, [Symbol("ob_ob_$(i)") for i in 1:N])
    append!(name, [Symbol("oa_oa_$(i)") for i in 1:N])
    append!(name, [Symbol("ab_ab_$(i)") for i in 1:N])
    append!(name, [Symbol("ab_ob_$(i)") for i in 1:N])
    append!(name, [Symbol("oa_ob_$(i)") for i in 1:N])
    append!(name, [Symbol("ab_oa_$(i)") for i in 1:N])

    df = DataFrame(results, name)
    CSV.write("./point_estimation_cc/"*@sprintf("etkfcc_with_est_%dpoint", n)*@sprintf("_add=%.1f.csv", add), df)
    
end



#=
for i in 0:40
    output_npoint_ETKFCC_with_true(0.6, i)
end
=#

for i in 0:40
    output_npoint_ETKFCC_with_est(0.6, i)
end
