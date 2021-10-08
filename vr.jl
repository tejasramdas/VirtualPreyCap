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
  #  xtraj = vcat(zeros(100), 100*ones(100), -100*(ones(300)))
    xtraj = zeros(500)
    ytraj = -50:.2:50
#    ytraj = 50*ones(500)
    #    ztraj = 20*ones(500)
    ztraj = zeros(500)
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
    row_res = 800
    col_res = 480
    env_fig = Figure(resolution=(row_res, col_res))
    limval = 100
    timenode = Node(1)
    para_trajectory = draw_para_trajectory()
    coords(t) = convert(Vector{Point3f0}, para_trajectory[t:t])
    # 8 px = 1mm, so fish is in a 25mm tank
    lim = (-limval, limval, -limval, limval, -limval, limval)
    # note perspectiveness variable is 0.0 for orthographic, 1.0 for perspective, .5 for intermediate
    # have to use SceneSpace for markerspace to get coords in dataunits. 
    env_axis = Axis3(env_fig[1,1], xtickcolor=black,
                     viewmode=:fit, aspect=(1,1,1), perspectiveness=0, protrusions=0, limits=lim)
    scatter!(env_axis, lift(t -> coords(t), timenode), markersize=3, markerspace=SceneSpace)
    # set rotation_center as the eyeposition b/c otherwise it rotates around the origin of the grid (i.e. the lookat)
    fish = cam3d!(env_axis.scene, eyeposition=Vec3f0(-limval, 0, 0), lookat=Vec3f0(0,0,0), fixed_axis=true, fov=100, rotation_center=:eyeposition)
    hidedecorations!(env_axis)
    hidespines!(env_axis)
    center!(env_axis.scene)
    # this is the only way to center it and get it to the point you want!
    translate_cam!(env_axis.scene, fish, Vec3f0(0, 0, fish.eyeposition[][1] + limval))
    translate_cam!(env_axis.scene, fish, Vec3f0(0, -fish.eyeposition[][3], 0))
    translate_cam!(env_axis.scene, fish, Vec3f0(fish.eyeposition[][2], 0, 0))


#    translate_cam!(env_axis.scene, fish, Vec3f0(0, 0, -50)) 
    display(env_fig)
    # for translate cam and rotate cam, the vectors are side to side, up and down, and into the screen. they aren't x, y, z, but are in the same units. angles are in rad. 
    for i in 1:500
        sleep(.01)
        timenode[] = i
        if i > 200
            translate_cam!(env_axis.scene, fish, Vec3f0(0, 0, -.3))
            rotate_cam!(env_axis.scene, fish, Vec3f0(0, .0015, 0))
        end
        
    end
    return env_axis, fish
end

# next steps are to use the code from SpikingInference distance model and re-create para trajectories inside the
# arena. try to create perspective from the fish's POV. 
