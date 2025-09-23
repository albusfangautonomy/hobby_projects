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

    qpos_adr = model.jnt_qposadr[jnt_id]
    # Set starting angle (e.g. 20 degrees)
    data.qpos[qpos_adr] = np.deg2rad(185)
    data.qpos[model.jnt_qposadr[j_slider]] = 0.5
    data.qvel[model.jnt_dofadr[jnt_id]] = 0.0
    data.qvel[model.jnt_dofadr[j_slider]] = 0.0

    
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        data.ctrl[a_cart] = +1.0
        for _ in range(50): mujoco.mj_step(model, data)
        print("Δx after +1N pulse:", data.qpos[model.jnt_qposadr[j_slider]] - 0.0)


if __name__ == "__main__":
    main()
