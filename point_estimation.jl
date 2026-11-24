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


"""
main
"""

function run_add_npoint(ρ, r, add, n, start_ensemble, eta, true_data)
    m = size(start_ensemble,2)
    r_inv = ones(Float64, N)
    r_inv[1:n] ./= r
    R_inv = Diagonal(r_inv)
    
    try
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
            y[1:n] .+= add * d[1:n]
            y .+= eta[:,k]

            d = y .- true_data[:,k]
            mse_y[t] = sum(d.*d) / Float64(N)

            Xa, stat = etkf(Xf_next, y)
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

        result = Vector{Float64}(undef, 6 + N*8)
        result[1] = add
        result[2] = ρ
        result[3] = r
        result[4] = sqrt(mean(mse_xa))
        result[5] = sqrt(mean(mse_xf))
        result[6] = sqrt(mean(mse_y))
        result[7:end] .= vec(stat_sum)
        
        return result
    catch
        result = Vector{Float64}(undef, 6 + N*8)
        result[1] = add
        result[2] = ρ
        result[3] = r
        result[4:end] .= zeros(Float64, 3 + N*8)
        return result
    end
end


function output_add_npoint(add, n)
    println("### start add=$(add), n=$(n)")
    ρs = [1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 3.00, 4.00, 5.00]
    rs = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.5, 1.7, 1.8, 1.9, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    eta = CSV.read("./eta.csv", DataFrame; header=false) |> Matrix |> transpose |> copy
    start_ensemble = CSV.read("./start_ensemble.csv", DataFrame; header=false) |> Matrix
    true_data = CSV.read("./true_data.csv", DataFrame; header=false) |> Matrix |> transpose |> copy

    
    results = zeros(Float64, length(ρs)*length(rs), 6 + N*8)

    Threads.@threads for j in  1:length(rs)
        Threads.@threads for i in  1:length(ρs)
            results[i+length(ρs)*(j-1),:] .= run_add_npoint(ρs[i], rs[j], add, n, start_ensemble, eta, true_data)
        end
    end

    #display(results)


    name = [:add, :multi_inf, :r, :rmse_a, :rmse_f, :rmse_o]
    append!(name, [Symbol("HBH_$(i)") for i in 1:N])
    append!(name, [Symbol("HAH_$(i)") for i in 1:N])
    append!(name, [Symbol("ob_ob_$(i)") for i in 1:N])
    append!(name, [Symbol("oa_oa_$(i)") for i in 1:N])
    append!(name, [Symbol("ab_ab_$(i)") for i in 1:N])
    append!(name, [Symbol("ab_ob_$(i)") for i in 1:N])
    append!(name, [Symbol("oa_ob_$(i)") for i in 1:N])
    append!(name, [Symbol("ab_oa_$(i)") for i in 1:N])

    df = DataFrame(results, name)
    CSV.write("./point_estimation/"*@sprintf("etkf_%dpoint", n)*@sprintf("_add=%.1f.csv", add), df)
    
end

#=
for k in 0:40
    output_add_npoint(0.6, k)
end
=#