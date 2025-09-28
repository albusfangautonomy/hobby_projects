import debugpy
debugpy.listen(("0.0.0.0", 5689))   
print("Waiting for debugger attach...")
debugpy.wait_for_client()
import numpy as np
import mujoco
import mujoco.viewer
import casadi as ca
import time
from acrobat_helper import linearize, angle_wrap_around_pi
from lqr import lqr, controllability, c2d, dlqr

debug = False

m1 = 3
m2 = 3
l1 = 0.5
l2 = 0.5
g = 9.81



Q = np.diag([20, 5, 1, 0.5])
R = np.diag([10])

Umin, Umax = -100.0, 100.0

def main():
    model = mujoco.MjModel.from_xml_path("acrobat.xml")
    data  = mujoco.MjData(model)

    dt = model.opt.timestep
    # Find hinge joint index
    link1_jnt_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "link1_joint")
    link2_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "link2_joint")
    torque_actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "torque")

    link1_qpos_adr = model.jnt_qposadr[link1_jnt_id]
    link2_qpos_adr = model.jnt_qposadr[link2_jnt_id]

    # Set starting angle (e.g. 20 degrees)
    data.qpos[link1_qpos_adr] = np.deg2rad(180.1)
    data.qpos[link2_qpos_adr] = np.deg2rad(0.1)

    data.qvel[model.jnt_dofadr[link1_jnt_id]] = 0.001
    data.qvel[model.jnt_dofadr[link2_jnt_id]] = 0.001

    z_eq = np.array([np.pi,0.0,0,0])

    A_lin, B_lin = linearize(m1, m2, l1, l2, g)
    # print("controllability rank = ", np.linalg.matrix_rank(np.array(controllability(A_lin, B_lin))))
    K = lqr(A_lin, B_lin, Q, R)
    A_d, B_d = c2d(A_lin,B_lin, dt)
    K_d, _ = dlqr(A_d, B_d, Q, R)
    T = 30.0
    steps = int(T/dt)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        sim_end = 20.0               # seconds of *simulation* time
        next_frame = time.time()
        print_timer = 0.0

        while viewer.is_running() and data.time < sim_end:
            # Read state values
            theta1  = data.qpos[model.jnt_qposadr[link1_jnt_id]]
            theta2  = data.qpos[model.jnt_qposadr[link2_jnt_id]]
            q1dot   = data.qvel[model.jnt_dofadr[link1_jnt_id]]
            q2dot   = data.qvel[model.jnt_dofadr[link2_jnt_id]]
         

            current_state = np.array([theta1, theta2, q1dot, q2dot])

            # Calculate error term
            e = np.array(current_state - z_eq)
            e[1] = angle_wrap_around_pi(e[1])
            e[0] = angle_wrap_around_pi(e[0])
            if debug:
                print(np.array(K).shape)
                print(e.shape)

            u_in = float(-(K @ e).squeeze())
            u_in = float(-(K_d @ e).squeeze())
            # Saturate to actuator limits
            u = max(Umin, min(Umax, u_in))
            data.ctrl[torque_actuator] = u

            # Print every 0.5 seconds of sim time
            if data.time - print_timer >= 0.5:
                print(f"t={data.time:.2f} | theta1={theta1:.3f}, theta_err={e[1]:.3f}, "
                    f"q1dot={q1dot:.3f}, q2dot={q2dot:.3f}, u={u:.3f}")
            
            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()
            next_frame += dt
            time.sleep(max(0.0, next_frame - time.time()))



if __name__ == "__main__":
    main()
