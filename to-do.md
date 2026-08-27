* preprocess tilt stacks before making a reconstruction (with warpylib function)
* remove one of the two network passes of the data fidelity loss to speed up processing
* reconstruct subvolumes with 3x oversampling, when refining a tomogram use a patch overlap of 0.5 
* write 1 refinement script that also does double network application
* check apodization after ctf multiplication
