import mujoco
import mujoco.viewer
import numpy as np

# Load the model
model = mujoco.MjModel.from_xml_path("cartpole.xml")
data = mujoco.MjData(model)

# Open a viewer window
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Simple control: random force
        data.ctrl[0] = np.random.uniform(-0.01, 0.01)
        # data.ctrl[0] = 0
        mujoco.mj_step(model, data)
        viewer.sync()
