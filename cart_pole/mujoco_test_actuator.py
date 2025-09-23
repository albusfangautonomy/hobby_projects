import mujoco
import mujoco.viewer
import numpy as np
from lqr import controllability, lqr
from cartpole_helper import angle_wrap_around_pi, linearize
import time

def main():
    model = mujoco.MjModel.from_xml_path("cartpole.xml")
    data  = mujoco.MjData(model)

    dt = model.opt.timestep
    # Find hinge joint index
    jnt_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
    j_slider = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "slider")
    a_cart = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cart_force")

    # qpos_adr = model.jnt_qposadr[jnt_id]
    # # Set starting angle (e.g. 20 degrees)
    # data.qpos[qpos_adr] = np.deg2rad(185)
    # data.qpos[model.jnt_qposadr[j_slider]] = 0.0
    # data.qvel[model.jnt_dofadr[jnt_id]] = 0.0
    # data.qvel[model.jnt_dofadr[j_slider]] = 0.0

    
    
        # Launch interactive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < 10:  # run 10 seconds
            step_start = time.time()

            # Apply control
            data.ctrl[a_cart] = +50.0

            # Step the simulation
            mujoco.mj_step(model, data)

            # Sync the viewer so it redraws
            viewer.sync()

            # Keep real time
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
