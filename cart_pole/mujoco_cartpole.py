import mujoco
import mujoco.viewer
import numpy as np
from lqr import controllability, lqr
model = mujoco.MjModel.from_xml_path("cartpole.xml")
data  = mujoco.MjData(model)
# Find hinge joint index
jnt_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
qpos_adr = model.jnt_qposadr[jnt_id]

# Set starting angle (e.g. 20 degrees)
data.qpos[qpos_adr] = np.deg2rad(180)

amp   = 0.0001        # N (well within your [-50, 50])
freq  = 0.005         # Hz
halfT = 0.5 / freq  # seconds

u     = +amp
tflip = 0.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        if data.time - tflip >= halfT:
            u = -u * 2
            tflip = data.time
            # Extract theta
            theta_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "theta")
            theta    = data.sensordata[theta_id]

            print(f"theta = {theta:.4f} rad")

        data.ctrl[0] = u
        mujoco.mj_step(model, data)
        viewer.sync()
