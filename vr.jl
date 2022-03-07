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

#gold = load(download("https://raw.githubusercontent.com/nidorx/matcaps/master/1024/E6BF3C_5A4719_977726_FCFC82.png"))


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
    ztraj = 30*ones(500)
    full_traj = zip(xtraj, ytraj, ztraj)
    return collect(full_traj)
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
    tail_θ, tail_xy = find_tail_angles(fishim, [1], [tailtop],
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
