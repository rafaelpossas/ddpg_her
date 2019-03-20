import pybullet as p
import time
import pybullet_data
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
objects = p.loadMJCF("assets/fetch/slide_table.xml")

puck_pos, puck_or = p.getBasePositionAndOrientation(2)
p.resetBasePositionAndOrientation(2, [0.45, 0.5, 0.425], puck_or)

print("{}. {}".format(puck_pos, puck_or))
count = 0
forward = 0
while(1):
    keys = p.getKeyboardEvents()

    for k, v in keys.items():
        if k == p.B3G_UP_ARROW and (v & p.KEY_WAS_TRIGGERED):
            forward = 1

    if forward:
        p.applyExternalForce(objects[2], -1, [0, forward, 0], puck_pos, flags=p.WORLD_FRAME)
        forward = 0

    p.stepSimulation()
    time.sleep(1. / 240.)
