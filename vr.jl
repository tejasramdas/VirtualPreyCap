using Spinnaker
using Images
using GLMakie

camlist = CameraList()
cam = camlist[0]
#triggersource!(cam, "Software")
#triggermode!(cam, "On")
#gain!(cam, 0)
#acquisitionmode!(cam, "Continuous")



# start!(cam)
#     #@info "Exposure set to $(expact/1e6)s"
#  #   trigger!(cam)
# #    saveimage(cam, joinpath(@__DIR__, "exposure_$(expval/1e6)s.png"), spinImageFileFmakeormat(6))
# image_from_cam = getimage(cam)
# im_arr = CameraImage(image_from_cam, UInt8)
# @info "Image saved"
# Images.save("/home/andrewbolton/VirtualPreyCap/im.png", im_arr)
# stop!(cam)

function draw_para_trajectory()
    xtraj = range(1, stop=500)
    ytraj = ones(500)
    ztraj = range(1, stop=500)
    full_traj = zip(xtraj, ytraj, ztraj)
    return collect(full_traj)
end



function make_VR_environment()
    black = RGBAf0(0, 0, 0, 0.0)
    row_res = 800
    col_res = 480
    env_fig = Figure(resolution=(row_res, col_res), figure_padding=-50)
    limval = 500
    timenode = Node(1)
    para_trajectory = draw_para_trajectory()
    coords(t) = para_trajectory[t]
    lim = (-limval, limval, -limval, limval, -limval, limval)
    # note perspectiveness variable is 0.0 for orthographic, 1.0 for perspective, .5 for intermediate
    env_axis = Axis3(env_fig[1,1], xtickcolor=black,
                     viewmode=:fit, aspect=(1,1,1), perspectiveness=0.0, protrusions=0, limits=lim)
    scatter!(env_axis, lift(t -> coords(t), timenode), color=:red, msize=5000) 
    display(env_fig)
    for i in 1:500
        sleep(.2)
        timenode[] = i
    end
    
end

# next steps are to use the code from SpikingInference distance model and re-create para trajectories inside the
# arena. try to create perspective from the fish's POV. 
