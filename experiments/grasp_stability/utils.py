import pybullet as pb

class Camera:
    def __init__(self, cameraResolution=[320, 240]):
        self.cameraResolution = cameraResolution

        camTargetPos = [0.5, 0, 0.05]
        camDistance = 0.6#0.4
        upAxisIndex = 2

        yaw = 0#90
        pitch = -30.0
        roll = 0

        fov = 60
        nearPlane = 0.01
        farPlane = 100

        self.viewMatrix = pb.computeViewMatrixFromYawPitchRoll(
            camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex
        )

        aspect = cameraResolution[0] / cameraResolution[1]

        self.projectionMatrix = pb.computeProjectionMatrixFOV(
            fov, aspect, nearPlane, farPlane
        )

    def get_image(self):
        img_arr = pb.getCameraImage(
            self.cameraResolution[0],
            self.cameraResolution[1],
            self.viewMatrix,
            self.projectionMatrix,
            shadow=1,
            lightDirection=[1, 1, 1],
            renderer=pb.ER_BULLET_HARDWARE_OPENGL,
        )

        rgb = img_arr[2]  # color data RGB
        dep = img_arr[3]  # depth data
        return rgb, dep


def get_forces(bodyA=None, bodyB=None, linkIndexA=None, linkIndexB=None):
    """
    get contact forces

    :return: normal force, lateral force
    """
    kwargs = {
        "bodyA": bodyA,
        "bodyB": bodyB,
        "linkIndexA": linkIndexA,
        "linkIndexB": linkIndexB,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    pts = pb.getContactPoints(**kwargs)

    totalNormalForce = 0
    totalLateralFrictionForce = [0, 0, 0]

    for pt in pts:
        totalNormalForce += pt[9]

        totalLateralFrictionForce[0] += pt[11][0] * pt[10] + pt[13][0] * pt[12]
        totalLateralFrictionForce[1] += pt[11][1] * pt[10] + pt[13][1] * pt[12]
        totalLateralFrictionForce[2] += pt[11][2] * pt[10] + pt[13][2] * pt[12]

    return totalNormalForce, totalLateralFrictionForce