using System;

namespace MJWarpDemo
{
    [Serializable]
    public sealed class ResponseEnvelope
    {
        public int protocol_version;
        public string type;
        public int request_id;
        public string error;
        public BackendInfo backend;
        public ModelSpec model_spec;
        public StateData state;
        public string episode_id;
        public string path;
        public string gpu;
        public bool recorded;
        public bool stopped;
        public int frame_count;
        public BenchmarkResult[] results;
        public ScenarioSummary[] scenarios;
    }

    [Serializable]
    public sealed class BackendInfo
    {
        public string name;
        public string mujoco_version;
        public string mujoco_warp_version;
        public string warp_version;
        public string gpu;
    }

    [Serializable]
    public sealed class ModelSpec
    {
        public string name;
        public string scenario_id;
        public string display_name;
        public string business_type;
        public string description;
        public string official_reference;
        public float physics_dt;
        public float control_dt;
        public int max_frames;
        public float goal_radius;
        public float[] camera_position;
        public float[] camera_look_at;
        public BodySpec[] bodies;
        public GeomSpec[] geoms;
    }

    [Serializable]
    public sealed class ScenarioSummary
    {
        public string scenario_id;
        public string display_name;
        public string business_type;
        public string description;
    }

    [Serializable]
    public sealed class BodySpec
    {
        public int id;
        public string name;
    }

    [Serializable]
    public sealed class GeomSpec
    {
        public int id;
        public string name;
        public int body_id;
        public string type;
        public float[] size;
        public float[] position;
        public float[] quaternion;
        public float[] rgba;
    }

    [Serializable]
    public sealed class StateData
    {
        public int frame_id;
        public double sim_time;
        public string scenario_id;
        public float[] qpos;
        public float[] qvel;
        public float[][] body_position;
        public float[][] body_quaternion;
        public float[][] body_external_wrench;
        public float[] action;
        public float reward;
        public bool terminated;
        public bool success;
        public float[] goal_position;
        public int task_stage;
        public float distance_to_goal;
        public ContactData contacts;
        public MetricsData metrics;
    }

    [Serializable]
    public sealed class ContactData
    {
        public int count;
        public bool[] valid;
        public int[][] geom_pair;
        public float[][] position;
        public float[][] normal;
        public float[] distance;
        public bool overflow;
    }

    [Serializable]
    public sealed class MetricsData
    {
        public int nworld;
        public int success_count;
        public float mean_reward;
        public float control_steps_per_second;
        public float physics_steps_per_second;
    }

    [Serializable]
    public sealed class BenchmarkResult
    {
        public int requested_nworld;
        public int actual_nworld;
        public bool fallback;
        public int steps;
        public double elapsed_seconds;
        public double physics_steps_per_second;
        public string error;
    }

    public sealed class CapturePayload
    {
        public int FrameId;
        public byte[] Rgb;
        public byte[] Depth;
        public byte[] Instance;
    }
}
