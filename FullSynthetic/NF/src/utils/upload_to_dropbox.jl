export upload_to_dropbox

"""
   upload_to_dropbox(sim_name::String)

   A function that uses `rclone` to upload the data and plots associated with a
   simulation—e.g., `./data/simulations` and `./plots/simulations`, (if exist) to
   Dropbox under the same project name and with the same directory structure as DrWatson.

   This function requires `rclone` to be configured, and it is not meant to be shared as
   a component of the project for reproducibility. Here, to exemplify its use, we are
   using `MyDropbox` as an example `rclone` remote name. This will not run properly if you
   do not configure `rclone` and use your remote name in this function.

*Input*:

- `sim_name`: Simulation name to upload

   """

function upload_to_dropbox(sim_name::String)

    cloud_data_path = joinpath(projectname(), "data", sim_name)
    cloud_plots_path = joinpath(projectname(), "plots", sim_name)

    # Upload data/sim_name
    if isdir(datadir(sim_name))
        run(`rclone copy --progress $(datadir(sim_name)) MyDropbox:$cloud_data_path`)
    end

    # Upload plots/sim_name
    if isdir(plotsdir(sim_name))
        run(`rclone copy --progress $(plotsdir(sim_name)) MyDropbox:$cloud_plots_path`)
    end

end
