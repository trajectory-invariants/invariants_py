import invariants_py.reparameterization as reparam
import invariants_py.class_frenetserret_calculation as FS_calculation

def calculate_invariants(trajectory, trajectory_type, progress_definition="arclength"):

    # Check if the trajectory type is valid
    if trajectory_type not in ["translation", "rotation"]:
        raise ValueError("Invalid trajectory type. The trajectory type must be either 'translation' or 'rotation'.")

    if trajectory_type == "translation":
        # different progress definition: {time, default: arclength, arcangle, screwbased}
        # Reparameterize the trajectory based on arclength
        trajectory_geom, arclength, arclength_n, nb_samples, stepsize = reparam.reparameterize_trajectory_arclength(trajectory)

        # Create an instance of the FrenetSerret_calc class
        FS_calculation_problem = FS_calculation.FrenetSerret_calc(window_len=nb_samples)

        # Calculate the invariants using the global method
        # TODO make a dictionary of the results from which invariants can be extracted
        result = FS_calculation_problem.calculate_invariants_global(trajectory_geom, stepsize=stepsize)
        invariants = result[0]

        return invariants, arclength

