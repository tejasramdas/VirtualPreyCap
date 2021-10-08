using Spinnaker
using Images
using GLMakie

#camlist = CameraList()
#cam = camlist[0]
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
    xtraj = range(1.0, stop=500)
    ytraj = ones(500)
    ztraj = range(1.0, stop=500)
    full_traj = zip(xtraj, ytraj, ztraj)
    return collect(full_traj)
end


function makie_test()
    f = Figure(resolution = (1200, 800), fontsize = 14)

    xs = LinRange(0, 10, 100)
    ys = LinRange(0, 10, 100)
    zs = [cos(x) * sin(y) for x in xs, y in ys]

    for (i, perspectiveness) in enumerate(LinRange(0, 1, 6))
        Axis3(f[fldmod1(i, 3)...], perspectiveness = perspectiveness,
            title = "$perspectiveness")

        surface!(xs, ys, zs)
    end
    display(f)
end


# note "msize" has been replaced by "markersize" 

function make_VR_environment()
    black = RGBAf0(0, 0, 0, 0.0)
    row_res = 1500
    col_res = 1500
    env_fig = Figure(resolution=(row_res, col_res))
    limval = 100
    timenode = Node(1)
    para_trajectory = draw_para_trajectory()
    coords(t) = convert(Vector{Point3f0}, para_trajectory[t:t])
    lim = (-limval, limval, -limval, limval, -limval, limval)
    # note perspectiveness variable is 0.0 for orthographic, 1.0 for perspective, .5 for intermediate
    env_axis = Axis3(env_fig[1,1], xtickcolor=black,
                     viewmode=:fit, aspect=(1,1,1), perspectiveness=0.0, protrusions=0, limits=lim)
    # scatter!(env_axis, lift(t -> coords(t), timenode), color=:black, markersize=10000)
    scatter!(env_axis, [Point3f0(0, 0, 20)], markersize=10000)
    fish = cam3d!(env_axis.scene)
    fish_origin = Vec3f0(-limval, 0, 0)
    fish_lookat_t0 = Vec3f0(0, 0, 0)
    update_cam!(env_axis.scene, fish, fish_origin, fish_lookat_t0)
    display(env_fig)
    # for i in 90:100
    #     sleep(.05)
    #     timenode[] = i
    # end
    return env_axis, fish
end

# next steps are to use the code from SpikingInference distance model and re-create para trajectories inside the
# arena. try to create perspective from the fish's POV. 
