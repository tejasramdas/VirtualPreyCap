using Gen
using GLMakie
using CairoMakie
using Distributions
using StatsBase

GLMakie.activate!()
@dist LabeledCategorical(labels, probs) = labels[categorical(probs)]

@gen function gainmodel(gain_stateₜ₋₁::String, prey_azimuth::Float64, swim_rotation::Float64)

    if gain_stateₜ₋₁₋ == "low"
        gain_transition_probs = [.7, .3, 0]
    elseif gain_stateₜ₋₁ == "mid"
        gain_transition_probs = [.05, .9, .05]
    elseif gain_stateₜ₋₁ == "high"
        gain_transition_probs = [0, .3, .7]
    end
    
    gain_stateₜ = { :gain_state } ~ LabeledCategorical(["low", "mid", "high"], visc_transition_probs)
    σ = .2
    
    if gain_stateₜ == "low"
        gain_swim = { :gain_swim } ~ normal(1.5, σ)
    elseif viscosity == "high"
        gain_swim = { :gain_swim } ~ normal(.5, σ)
    else
        gain_swim = { :gain_swim } ~ normal(1, σ)
    end

    new_prey_azimuth = { :new_prey_azimuth } ~ normal(prey_azimuth - (gain_swim * swim_rotation), σ / 2)
    
    return gain_stateₜ, prey_azimuth
end

function plot_gaussian_mixture()
    x = range(-3, stop=3, length=1000)
    lowgain = Normal(.5, .2)
    midgain = Normal(1, .2)
    highgain = Normal(1.5, .2)
    fig = Figure()
    ax = fig[1,1] = Axis(fig)
    xlims!(ax, (-1, 3))
    lines!(ax, x, [pdf(lowgain, xval) for xval in x])
    lines!(ax, x, [pdf(midgain, xval) for xval in x])
    lines!(ax, x, [pdf(highgain, xval) for xval in x])
    display(fig)
    return fig
end


function dendrite_w_threshold(n_excitatory_synapses, n_inhibitory_synapses, sim_duration_ms)
    pₑ = .05
    threshold = 20
    res = 1
    lfp_freq = 8
    # repeats every 125 ms
    lfp_period = 1000 * (1 / lfp_freq)
    lfp_wave = [sin(x) for x in 1:sim_duration_ms]
    summed_excitatory = zeros(sim_duration_ms)
    summed_inhibitory = zeros(sim_duration_ms)
    for n in 1:n_excitatory_synapses
        excitatory = [1*bernoulli(pₑ) for i in 1:sim_duration_ms]
        summed_excitatory += excitatory
    end
    for n in 1:n_inhibitory_synapses
        inhibitory = [-1*bernoulli(pₑ) for i in 1:sim_duration_ms]
        summed_inhibitory += inhibitory
    end
    integrated_soma = map(f -> f < 0 ? 0 : f, summed_excitatory + summed_inhibitory)
    fig = hist(integrated_soma)
    display(fig)
    countmap(integrated_soma)
    return integrated_soma
end

# RATE CAN BE NUMBER OF SYNAPSES EACH W SAME RATE. LEARNING IS SYNAPSES RETRACTING AND REMODELING.
# What are the interesting features of neurons? They dont go both ways. I.e. a logical relationship cannot be represented
# by an action potential in the opposite direction.

# CHECK RICE FOR EXPLANATION ON WHEN TO USE POISSON AND WHY. 


    
    


