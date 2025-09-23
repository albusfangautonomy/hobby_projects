import debugpy
debugpy.listen(("0.0.0.0", 5689))   # open debugger server on port 5678
print("Waiting for debugger attach...")
debugpy.wait_for_client()

import mujoco
import mujoco.viewer
import numpy as np
from lqr import controllability, lqr
from cartpole_helper import angle_wrap_around_pi, linearize
import time

mp = 0.316
mc = 0.5
l = 0.2
g = 9.81

Q = np.diag([0.1, 5, 0.1, 0.05])
R = np.diag([1.0])

Umin, Umax = -10.0, 10.0


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

    z_eq = np.array([0.0,np.pi,0,0])

    A_lin, B_lin = linearize(mp, mc, l, g)
    K = lqr(A_lin, B_lin, Q, R)

    T = 20.0
    steps = int(T/dt)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        sim_end = 20.0               # seconds of *simulation* time
        next_frame = time.time()
        print_timer = 0.0

        while viewer.is_running() and data.time < sim_end:
        
        # for k in range(steps):
            # Read joint states directly (simpler than sensors)
            x      = data.qpos[model.jnt_qposadr[j_slider]]
            theta  = data.qpos[model.jnt_qposadr[jnt_id]]
            xdot   = data.qvel[model.jnt_dofadr[j_slider]]
            angular_vel   = data.qvel[model.jnt_dofadr[jnt_id]]

            current_state = np.array([x, theta, xdot, angular_vel])
            # Calculate error term
            e = np.array(current_state - z_eq)
            e[1] = angle_wrap_around_pi(e[1])
            # print(np.array(K).shape)
            # print(e.shape)

            u_in = float(-(K @ e).squeeze())
            
            # Saturate to actuator limits
            u = max(Umin, min(Umax, u_in))
            data.ctrl[a_cart] = u

                    # Print every 0.5 seconds of sim time
            if data.time - print_timer >= 0.5:
                print(f"t={data.time:.2f} | x={x:.3f}, theta_err={e[1]:.3f}, "
                    f"xdot={xdot:.3f}, angular_vel={angular_vel:.3f}, u={u:.3f}")
            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()
            next_frame += dt
            time.sleep(max(0.0, next_frame - time.time()))
            # if k % int(0.5 / dt) == 0:
            #     print(
            #         f"t={k*dt:5.2f}  x={x:+.3f}  th_err={e[1]:+.3f}  xdot={xdot:+.3f}  tdot={angular_vel:+.3f}  u={u:+.2f}"
            #     )

if __name__ == "__main__":
    main()
