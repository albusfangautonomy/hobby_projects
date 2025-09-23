import mujoco
import time
import mujoco.viewer
def main():
    model = mujoco.MjModel.from_xml_path("acrobat.xml")
    data  = mujoco.MjData(model)

    dt = model.opt.timestep
    # Find hinge joint index
    link1_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "link1_joint")
    link2_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "link2_joint")
    torque = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "torque")

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
            data.ctrl[torque] = +0.5

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
