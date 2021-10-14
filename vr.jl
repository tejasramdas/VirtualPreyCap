#using Spinnaker
using Images
using GLMakie
using GeometryBasics
using LinearAlgebra

#camlist = CameraList()
#cam = camlist[0]
#triggersource!(cam, "Software")
#triggermode!(cam, "On")
#gain!(cam, 0)
#acquisitionmode!(cam, "Continuous")

gold = load(download("https://raw.githubusercontent.com/nidorx/matcaps/master/1024/E6BF3C_5A4719_977726_FCFC82.png"))


py"""

import json
import pandas as pd
import sppl.compilers.spn_to_dict as spn_to_dict
from sppl.transforms import Identity as I
from collections import OrderedDict

df = pd.read_csv('/Users/nightcrawler/inferenceql.auto-modeling/data/nullified.csv')

def loader():
    with open('/Users/nightcrawler/inferenceql.auto-modeling/data/sppl/merged.json') as f:
        spn_dict = json.load(f)
    model = spn_to_dict.spn_from_dict(spn_dict)
    return model

def generate(model, N):
    samples = model.sample(N)
    return pd.DataFrame(
        [
            {k.__str__():v
             for k,v in sample.items()
             if ('_cluster' not in k.__str__()) and k.__str__()!='child'
            }
        for sample in samples
    ])[df.columns]

"""
model = py"loader"()
py"generate"(model, 1)

function generate_bout(mod)
    bout_sample = py"generate"(mod, 1)
    [convert(Float64, bout_sample[bout_variable]) for bout_variable in ["BoutAz", "BoutYaw", "BoutDistance"]]
end    


# start!(cam)
#     #@info "Exposure set to $(expact/1e6)s"
#  #   trigger!(cam)
# #    saveimage(cam, joinpath(@__DIR__, "exposure_$(expval/1e6)s.png"), spinImageFileFmakeormat(6))
# image_from_cam = getimage(cam)
# im_arr = CameraImage(image_from_cam, UInt8)
# @info "Image saved"
# Images.save("/home/andrewbolton/VirtualPreyCap/im.png", im_arr)
# stop!(cam)

roundint(x) = round(Int, x)
tail_circle_res = 128
zero_to_pi = 0:-π/(tail_circle_res / 2):-π
pi_to_zero = π:-π/(tail_circle_res / 2):0
arc_index_to_angle(x) = vcat(zero_to_pi[1:end-1], pi_to_zero[1:end-1])[x]


fishimage = load("embedded_fish_bent_right.png")
fishimage = convert(Matrix{Gray}, fishimage)
fishimage = convert(Matrix{UInt8}, fishimage * 255)




function draw_para_trajectory()
    xtraj = zeros(500)
    ytraj = -50:.2:50
    ztraj = 20*ones(500)
    full_traj = zip(xtraj, ytraj, ztraj)
    return collect(full_traj)
end



# TODO:
# decide whether to use textured ground. may be a research project in itself.
# add real trajectories with real choices.
# install python deps for IQL onto the rig computer. make sure you can
# get pycall to import the correct conda environment. 
function make_ground_mesh(lim, radius)
    # make vertices the edge of a circle with zero z coords
    # and a center coord that connects all vertices with its next door neighbor and the center for faces.
    floor_circle = Circle(Point2f(0, 0), radius)
    vertices = Point{3, Float64}[(c[1], c[2], -lim) for c in coordinates(floor_circle)]
    push!(vertices, Point3f(0, 0, -lim))
    faces = TriangleFace{Cint}[[i, i+1, length(vertices)] for i in 1:length(vertices)-2]
    rendered_mesh = GeometryBasics.Mesh(vertices, faces)
    return uv_normal_mesh(rendered_mesh)
end

function make_VR_environment()
    black = RGBAf0(0, 0, 0, 0.0)
    row_res = 800
    col_res = 480
    env_fig = Figure(resolution=(row_res, col_res))
    limval = 100
    mesh_cover = load("stone.png")
    timenode = Node(1)
    para_trajectory = draw_para_trajectory()
    coords(t) = convert(Vector{Point3f0}, para_trajectory[t:t])
    # 8 px = 1mm, so fish is in a 25mm tank
    lim = (-limval, limval, -limval, limval, -limval, limval)
    # perspectiveness variable is 0.0 for orthographic, 1.0 for perspective, .5 for intermediate
    # have to use SceneSpace for markerspace to get coords in dataunits.
    # note "msize" has been replaced by "markersize" 
    env_axis = Axis3(env_fig[1,1], xtickcolor=black,
                     viewmode=:fit, aspect=(1,1,1), perspectiveness=0.5, protrusions=0, limits=lim)
    mesh!(env_axis, make_ground_mesh(limval, 10000), color=mesh_cover, shading=false)
    scatter!(env_axis, lift(t -> coords(t), timenode), markersize=5, markerspace=SceneSpace)
    # set rotation_center as the eyeposition b/c otherwise it rotates around the origin of the grid (i.e. the lookat)
    fish = cam3d!(env_axis.scene, eyeposition=Vec3f0(-limval, 0, 0), lookat=Vec3f0(0,0,0), fixed_axis=true, fov=100, rotation_center=:eyeposition)
  #  hidedecorations!(env_axis)
  #  hidespines!(env_axis)
    center!(env_axis.scene)
    # this is the only way to center it and get it to the point you want!
    translate_cam!(env_axis.scene, fish, Vec3f0(0, 0, fish.eyeposition[][1] + limval))
    translate_cam!(env_axis.scene, fish, Vec3f0(0, -fish.eyeposition[][3], 0))
    translate_cam!(env_axis.scene, fish, Vec3f0(fish.eyeposition[][2], 0, 0))
    display(env_fig)
    # for translate cam and rotate cam, the vectors are side to side, up and down, and into the screen. they aren't x, y, z, but are in the same units. angles are in rad.
    # here go for the tail angle.
    # write an output function here to activate the pyboard triggering run.
    # translate cam 3rd variable is translation in X. its negative to go forward.
    # rotate cam yaw angle is the second variable. 

    stop_experiment = false
    toggle_experiment() = stop_experiment ? stop_experiment = false : stop_experiment = true

    on(events(env_fig).keyboardbutton) do buttonpress
        if buttonpress.action == Keyboard.press
            buttonpress.key == Keyboard.down && toggle_experiment()
        end
    end
    
    fish_position = []
    fish_rotation = []
    i = 1
    while(!stop_experiment)
        timenode[] = i
        i += 1
        sleep(.01)
        dyaw = 0
        if i % 50 == 0
#            dyaw = rand(range(-.1, .1, step=.001))
 #           b_az = -.5
  #          b_dist = 3
            b_az, dyaw, b_dist = generate_bout(model)
            uvec = [cos(b_az), sin(b_az)]
            b_vec = b_dist * uvec
            translate_cam!(env_axis.scene, fish, Vec3f0(b_vec[2], 0, -b_vec[1]))
            rotate_cam!(env_axis.scene, fish, Vec3f0(0, -dyaw, 0))
        end
        push!(fish_position, fish.eyeposition[][1:2])
        push!(fish_rotation, dyaw)
        if i == 500
            i = 1
        end
    end
    return env_axis, fish, fish_position, cumsum(fish_rotation)
end


# define type of camera here, have it take an image and display it. 

function input_tail_boundaries(fishfig, fishax)
    top_and_bottom_coords = []
    on(events(fishfig).mousebutton) do buttonpress
        if buttonpress.action == Mouse.press
            push!(top_and_bottom_coords, mouseposition(fishax.scene))
        end
    end
    query_enter() = length(top_and_bottom_coords) == 2
    # will either stop when you press enter or when 200 seconds have passed. 
    timedwait(query_enter, 2000.0)
    return top_and_bottom_coords
end


# Circle uses 64 coords to define the circle.
# 500x500 image, get ~4ms for a gaussian filter, implemented with imfilter(im, Kernel.gaussian(std)).
# 700x700 is a 10ms gaussian.

# circshift can cycle the array. 

function tailtracker(fishim)
    fishfig, fishax = image(fishim)
    fishax.aspect = DataAspect()
    fishax.title = "I'm a fish broseph"
    display(fishfig)
    tailtop, tailbottom = input_tail_boundaries(fishfig, fishax)
    println(tailtop, tailbottom)
    tail_length = norm(tailtop - tailbottom)
    numsegs = 20
    seg_length = tail_length / numsegs
    # filter image here if you want.
    @time tail_θ, tail_xy = find_tail_angles(fishim, [1], [tailtop],
                                             numsegs, (Point2(tailtop...), seg_length))
    [poly!(fishax, Circle(Point2f(xy...), 5)) for xy in tail_xy[6:end]]
    return tail_θ[7:end], tail_xy[6:end]
end


# this is all correct now except the indicies are inverted -- i think
# positive should be positive and neg should be neg...otherwise mag and dot placement is right.


# Circle object with no shift contains ordered coordinates. They start at the bottom of the circle and move
# clockwise around the circle. subtraction of the sum of all tail angles aligns the current circle to the
# orientation of the fish, and puts the center of the arc bottom at index 1. you have to subtract
# the length of tailangles b/c circshift operates on a 0 index (e.g. cirshift of 0 is no shift); the bottom of the circle should be zero shifted, but occupies index 1 in the arc. 

function find_tail_angles(im, tailangles, tailpos, numsegs, circle_params)
    if numsegs == 0
        return tailangles, tailpos
    else
        arc = collect(coordinates(Circle(circle_params...), tail_circle_res))
        # shifts the center bottom of the circle (i.e. the vector pointing from the circle center along the tail) to the first element. 
        shifted_arc = circshift(arc, -1*(sum(tailangles) - length(tailangles)))
        arc_bottom_half = vcat(shifted_arc[Int(tail_circle_res * 3 / 4) + 1:end],
                               shifted_arc[1:Int(tail_circle_res / 4)])
        # FOR DEBUGGING -- THIS WILL PLOT EACH ARC. 
        # cmap = range(colorant"skyblue2", stop=colorant"navyblue", length=length(arc_bottom_half))
        # for (i, coord) in enumerate(arc_bottom_half)
        #     poly!(fishax, Circle(Point2f(coord...), 5), color=cmap[i])
        # end
        imvals_on_arc = map(f -> im[roundint.(f)...], arc_bottom_half)
        minval = findmin(imvals_on_arc)
        next_tailpoint = arc_bottom_half[minval[2]] # this is the circle coordinate with the max or min value.
        ta_curr = findfirst(isequal(next_tailpoint), shifted_arc)
        find_tail_angles(im, vcat(tailangles, ta_curr), vcat(tailpos, [next_tailpoint]),
                         numsegs-1, (Point2(next_tailpoint...), circle_params[2]))
    end
end





# probably want one thread to read camera images and extract tail angles
# have a second thread that waits for tail angles and updates the VR accordingly.

# either way you have to set the top and bottom of the tail. first press t, then mouseclick, then b the mouseclick. 
