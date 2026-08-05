using UnityEngine;

namespace MJWarpDemo
{
    public static class MjWarpCoordinates
    {
        public static Vector3 Position(float[] mujocoPosition)
        {
            if (mujocoPosition == null || mujocoPosition.Length < 3)
                return Vector3.zero;
            return new Vector3(mujocoPosition[0], mujocoPosition[2], mujocoPosition[1]);
        }

        // MuJoCo wire order is (w, x, y, z). Unity stores (x, y, z, w).
        public static Quaternion Rotation(float[] mujocoQuaternion)
        {
            if (mujocoQuaternion == null || mujocoQuaternion.Length < 4)
                return Quaternion.identity;
            return new Quaternion(
                mujocoQuaternion[1],
                mujocoQuaternion[3],
                mujocoQuaternion[2],
                -mujocoQuaternion[0]);
        }

        public static Vector3 Size(float x, float y, float z)
        {
            return new Vector3(x, z, y);
        }
    }
}
