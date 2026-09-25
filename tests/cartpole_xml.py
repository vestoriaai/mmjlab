CARTPOLE_XML = """
<mujoco model="cartpole">
  <option timestep="0.01"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body name="rail" pos="0 0 0.1">
      <joint name="slider" type="slide" axis="1 0 0" range="-2 2"/>
      <geom name="cart" type="box" size="0.15 0.1 0.05" mass="1"/>
      <body name="pole" pos="0 0 0.05">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1"/>
        <geom name="pole" type="capsule" fromto="0 0 0 0 0 0.5" size="0.02" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor joint="slider" gear="10"/></actuator>
</mujoco>
"""
