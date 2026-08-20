using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace MJWarpDemo
{
    public sealed class RendererBinding
    {
        public Renderer Renderer;
        public Material OriginalMaterial;
        public Material InstanceMaterial;
        public int InstanceId;
        public Color BaseColor;
    }

    public sealed class MjWarpVisualizer : IDisposable
    {
        private readonly Dictionary<int, Transform> bodyTransforms = new Dictionary<int, Transform>();
        private readonly Dictionary<string, Transform> bodyNameTransforms = new Dictionary<string, Transform>();
        private readonly List<RendererBinding> rendererBindings = new List<RendererBinding>();
        private readonly List<Material> ownedMaterials = new List<Material>();
        private readonly List<Mesh> ownedMeshes = new List<Mesh>();
        private GameObject root;
        private Transform goalTransform;

        public IReadOnlyList<RendererBinding> RendererBindings => rendererBindings;

        public void Build(ModelSpec spec)
        {
            Dispose();
            root = new GameObject("MJWarp Visual Proxies");
            foreach (BodySpec body in spec.bodies)
            {
                var bodyObject = new GameObject($"Body {body.id}: {body.name}");
                bodyObject.transform.SetParent(root.transform, false);
                bodyTransforms[body.id] = bodyObject.transform;
                bodyNameTransforms[body.name] = bodyObject.transform;
            }

            foreach (GeomSpec geom in spec.geoms)
            {
                if (!bodyTransforms.TryGetValue(geom.body_id, out Transform body))
                    continue;
                GameObject visual = CreateGeom(geom);
                if (visual == null)
                    continue;
                visual.name = $"Geom {geom.id}: {geom.name}";
                visual.transform.SetParent(body, false);
                visual.transform.localPosition = MjWarpCoordinates.Position(geom.position);
                visual.transform.localRotation = MjWarpCoordinates.Rotation(geom.quaternion);
                Collider collider = visual.GetComponent<Collider>();
                if (collider != null)
                    UnityEngine.Object.Destroy(collider);

                Renderer renderer = visual.GetComponent<Renderer>();
                if (renderer == null)
                    continue;
                Material rgbMaterial = CreateRgbMaterial(geom.rgba);
                Material instanceMaterial = CreateInstanceMaterial(geom.id + 1);
                renderer.sharedMaterial = rgbMaterial;
                rendererBindings.Add(new RendererBinding
                {
                    Renderer = renderer,
                    OriginalMaterial = rgbMaterial,
                    InstanceMaterial = instanceMaterial,
                    InstanceId = geom.id + 1,
                    BaseColor = rgbMaterial.color,
                });
            }

            CreateGoalMarker(spec.goal_radius > 0f ? spec.goal_radius : 0.06f);
        }

        private GameObject CreateGeom(GeomSpec geom)
        {
            float[] size = geom.size ?? Array.Empty<float>();
            switch (geom.type)
            {
                case "plane":
                {
                    GameObject plane = GameObject.CreatePrimitive(PrimitiveType.Cube);
                    float x = size.Length > 0 ? size[0] * 2f : 2f;
                    float z = size.Length > 1 ? size[1] * 2f : 2f;
                    plane.transform.localScale = new Vector3(x, 0.02f, z);
                    plane.transform.localPosition = new Vector3(0f, -0.01f, 0f);
                    return plane;
                }
                case "box":
                {
                    GameObject box = GameObject.CreatePrimitive(PrimitiveType.Cube);
                    box.transform.localScale = MjWarpCoordinates.Size(size[0] * 2f, size[1] * 2f, size[2] * 2f);
                    return box;
                }
                case "sphere":
                {
                    GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                    sphere.transform.localScale = Vector3.one * size[0] * 2f;
                    return sphere;
                }
                case "cylinder":
                {
                    GameObject cylinder = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                    cylinder.transform.localScale = new Vector3(size[0] * 2f, size[1], size[0] * 2f);
                    return cylinder;
                }
                case "capsule":
                {
                    GameObject capsule = GameObject.CreatePrimitive(PrimitiveType.Capsule);
                    float radius = size[0];
                    float halfLength = size.Length > 1 ? size[1] : radius;
                    capsule.transform.localScale = new Vector3(radius * 2f, halfLength + radius, radius * 2f);
                    return capsule;
                }
                case "mesh":
                {
                    if (geom.vertices == null || geom.triangles == null || geom.vertices.Length == 0)
                        return null;
                    var visual = new GameObject("MJWarp Mesh");
                    var filter = visual.AddComponent<MeshFilter>();
                    visual.AddComponent<MeshRenderer>();
                    var mesh = new Mesh { name = $"MJWarp Mesh {geom.id}" };
                    if (geom.vertices.Length > 65535)
                        mesh.indexFormat = IndexFormat.UInt32;
                    var vertices = new Vector3[geom.vertices.Length];
                    for (int index = 0; index < vertices.Length; index++)
                        vertices[index] = MjWarpCoordinates.Position(geom.vertices[index]);
                    var triangles = (int[])geom.triangles.Clone();
                    // MuJoCo -> Unity includes a handedness reflection, so flip triangle winding.
                    for (int index = 0; index + 2 < triangles.Length; index += 3)
                    {
                        int swap = triangles[index + 1];
                        triangles[index + 1] = triangles[index + 2];
                        triangles[index + 2] = swap;
                    }
                    mesh.vertices = vertices;
                    mesh.triangles = triangles;
                    mesh.RecalculateNormals();
                    mesh.RecalculateBounds();
                    filter.sharedMesh = mesh;
                    ownedMeshes.Add(mesh);
                    return visual;
                }
                default:
                    return null;
            }
        }

        public Transform FindBodyTransform(string bodyName)
        {
            return bodyName != null && bodyNameTransforms.TryGetValue(bodyName, out Transform body)
                ? body
                : null;
        }

        private Material CreateRgbMaterial(float[] rgba)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
            var material = new Material(shader) { name = "MJWarp RGB Material" };
            Color color = rgba != null && rgba.Length >= 4
                ? new Color(rgba[0], rgba[1], rgba[2], rgba[3])
                : Color.gray;
            if (material.HasProperty("_BaseColor"))
                material.SetColor("_BaseColor", color);
            material.color = color;
            ownedMaterials.Add(material);
            return material;
        }

        private Material CreateInstanceMaterial(int instanceId)
        {
            Shader shader = Shader.Find("MJWarp/InstanceId");
            if (shader == null)
                throw new InvalidOperationException("缺少 MJWarp/InstanceId 实例分割 Shader");
            var material = new Material(shader) { name = $"Instance {instanceId}" };
            // Use a Vector property, not a Color property: Unity color-space conversion
            // would otherwise quantize small IDs such as 1/255 back to zero.
            var encoded = new Vector4((instanceId & 0xff) / 255f, ((instanceId >> 8) & 0xff) / 255f, 0f, 1f);
            material.SetVector("_ObjectIdColor", encoded);
            ownedMaterials.Add(material);
            return material;
        }

        private void CreateGoalMarker(float radius)
        {
            GameObject goal = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            goal.name = "Goal Target";
            goal.transform.SetParent(root.transform, false);
            float diameter = radius * 2f;
            goal.transform.localScale = new Vector3(diameter, 0.003f, diameter);
            Collider collider = goal.GetComponent<Collider>();
            if (collider != null)
                UnityEngine.Object.Destroy(collider);
            Renderer renderer = goal.GetComponent<Renderer>();
            Material rgb = CreateRgbMaterial(new[] { 0.16f, 0.9f, 0.32f, 1f });
            Material instance = CreateInstanceMaterial(65534);
            renderer.sharedMaterial = rgb;
            rendererBindings.Add(new RendererBinding
            {
                Renderer = renderer,
                OriginalMaterial = rgb,
                InstanceMaterial = instance,
                InstanceId = 65534,
                BaseColor = rgb.color,
            });
            goalTransform = goal.transform;
        }

        public void ApplyState(StateData state)
        {
            if (state?.body_position == null || state.body_quaternion == null)
                return;
            int count = Math.Min(state.body_position.Length, state.body_quaternion.Length);
            for (int bodyId = 0; bodyId < count; bodyId++)
            {
                if (!bodyTransforms.TryGetValue(bodyId, out Transform body))
                    continue;
                body.position = MjWarpCoordinates.Position(state.body_position[bodyId]);
                body.rotation = MjWarpCoordinates.Rotation(state.body_quaternion[bodyId]);
            }
            if (goalTransform != null)
                goalTransform.position = MjWarpCoordinates.Position(state.goal_position) + Vector3.up * 0.004f;
        }

        public void RandomizeAppearance(int seed)
        {
            var random = new System.Random(seed);
            foreach (RendererBinding binding in rendererBindings)
            {
                if (binding.InstanceId == 65534)
                    continue;
                Color baseColor = binding.BaseColor;
                float scale = 0.85f + (float)random.NextDouble() * 0.3f;
                Color varied = new Color(
                    Mathf.Clamp01(baseColor.r * scale),
                    Mathf.Clamp01(baseColor.g * scale),
                    Mathf.Clamp01(baseColor.b * scale),
                    baseColor.a);
                if (binding.OriginalMaterial.HasProperty("_BaseColor"))
                    binding.OriginalMaterial.SetColor("_BaseColor", varied);
                binding.OriginalMaterial.color = varied;
            }
        }

        public void Dispose()
        {
            bodyTransforms.Clear();
            bodyNameTransforms.Clear();
            rendererBindings.Clear();
            foreach (Material material in ownedMaterials)
            {
                if (material != null)
                    UnityEngine.Object.Destroy(material);
            }
            ownedMaterials.Clear();
            foreach (Mesh mesh in ownedMeshes)
            {
                if (mesh != null)
                    UnityEngine.Object.Destroy(mesh);
            }
            ownedMeshes.Clear();
            if (root != null)
                UnityEngine.Object.Destroy(root);
            root = null;
            goalTransform = null;
        }
    }
}
