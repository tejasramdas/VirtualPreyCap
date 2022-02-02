using Spinnaker
using Images
using GeometryBasics
using LinearAlgebra
using GLMakie

function initialize_camera()
    camlist = CameraList()
    cam = camlist[0]
    start!(cam)
    return cam
end

function get_images(cam::Spinnaker.Camera, num_images::Int64)
    for i in 1:num_images
        image_from_cam = getimage(cam)
        im_arr = CameraImage(image_from_cam, UInt8)
        @info "Image saved"
        Images.save("/home/andrewbolton/VirtualPreyCap/im.png", im_arr)
    end
    stop!(cam)        
end


# BRING THESE FUNCTIONS INTO THE VR PROGRAM. EMBED A FISH AND EXTRACT TAIL ANGLES FROM LIVE IMAGES. PLOT THEM IN REAL TIME.


## trigger!(cam)







#triggersource!(cam, "Software")
#triggermode!(cam, "On")
#gain!(cam, 0)
#acquisitionmode!(cam, "Continuous")
# #    saveimage(cam, joinpath(@__DIR__, "exposure_$(expval/1e6)s.png"), spinImageFileFmakeormat(6))
