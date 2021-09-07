using Gen
using GLMakie

@dist LabeledCategorical(labels, probs) = labels[categorical(probs)]

@gen function gainmodel(prev_visc::String)

    if prev_visc == "low"
        visc_transition_probs = [.5, .5, 0]
    elseif prev_visc == "mid"
        visc_transition_probs = [.05, .95, .05]
    elseif prev_visc == "high"
        visc_transition_probs = [0, .5, .5]
    end
    
    viscosity = { :viscosity } ~ LabeledCategorical(["low", "mid", "high"], visc_transition_probs)
        
    if viscosity == "low"
        gain = { :gain } ~ normal(1.5, .1)
    elseif viscosity == "high"
        gain = { :gain } ~ normal(.5, .1)
    else
        gain = { :gain } ~ normal(1, .1)
    end
    
    return gain 
end


function move_prey(prey_azimuth::Float64, prev_visc::String, swim_rotation::Float64)
    (trace, _) = Gen.generate(gainmodel, (prev_visc,))
    gain = get_retval(trace)
    new_prey_azimuth = prey_azimuth - (gain * swim_rotation)
    return new_prey_azimuth
end

