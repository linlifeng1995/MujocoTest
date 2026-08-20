using System;
using System.Collections;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace MJWarpDemo.Tests
{
    public sealed class EndToEndPlayModeTests
    {
        [UnityTest]
        [Timeout(900000)]
        public IEnumerator PandaPilotEpisodesRoundTrip()
        {
            var cancellation = new CancellationTokenSource(TimeSpan.FromMinutes(14));
            var client = new MjWarpClient();
            Task connect = client.ConnectAsync("127.0.0.1", 8765, 10000, cancellation.Token);
            yield return Await(connect);

            yield return RunPandaEpisode(client, "panda_pick_place", "unity_panda_pick_place_seed0", cancellation.Token);
            yield return RunPandaEpisode(client, "panda_peg_insert", "unity_panda_peg_insert_seed0", cancellation.Token);

            Task<ResponseEnvelope> shutdownTask = client.SendAsync("shutdown", null, cancellation.Token);
            yield return Await(shutdownTask);
            client.Dispose();
            cancellation.Dispose();
        }

        [UnityTest]
        [Timeout(120000)]
        public IEnumerator PhysicsRenderCaptureAndHdf5RoundTrip()
        {
            var cancellation = new CancellationTokenSource(TimeSpan.FromSeconds(110));
            var client = new MjWarpClient();
            var visualizer = new MjWarpVisualizer();
            Camera camera = null;
            MultiModalCapture capture = null;

            Task connect = client.ConnectAsync("127.0.0.1", 8765, 10000, cancellation.Token);
            yield return Await(connect);

            Task<ResponseEnvelope> helloTask = client.SendAsync("hello", new { client = "UnityPlayModeTest" }, cancellation.Token);
            yield return Await(helloTask);
            ResponseEnvelope hello = helloTask.Result;
            Assert.That(hello.backend.name, Is.EqualTo("MJWarp"));
            Assert.That(hello.model_spec.scenario_id, Is.EqualTo("planar_push"));
            Assert.That(hello.scenarios.Length, Is.EqualTo(6));
            Assert.That(hello.model_spec.geoms.Length, Is.GreaterThan(0));

            camera = new GameObject("Test Capture Camera").AddComponent<Camera>();
            camera.transform.position = new Vector3(-0.02f, 0.86f, -0.88f);
            camera.transform.LookAt(new Vector3(-0.04f, 0.035f, 0f));
            camera.nearClipPlane = 0.03f;
            camera.farClipPlane = 4f;
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = Color.black;
            visualizer.Build(hello.model_spec);
            capture = new MultiModalCapture(camera);
            capture.SetBindings(visualizer.RendererBindings);

            Task<ResponseEnvelope> resetTask = client.SendAsync("reset", new { seed = 11, policy = "expert", nworld = 1 }, cancellation.Token);
            yield return Await(resetTask);
            visualizer.ApplyState(resetTask.Result.state);

            Task<ResponseEnvelope> startTask = client.SendAsync(
                "record_start",
                new { episode_id = "unity_playmode_smoke", seed = 11, policy = "expert", image_width = 320, image_height = 240 },
                cancellation.Token);
            yield return Await(startTask);

            Task<CapturePayload> initialCaptureTask = capture.CaptureAsync(resetTask.Result.state.frame_id);
            yield return Await(initialCaptureTask);
            CapturePayload initialImages = initialCaptureTask.Result;
            Task<ResponseEnvelope> initialUploadTask = client.SendAsync(
                "capture",
                new
                {
                    initial = true,
                    frame_id = resetTask.Result.state.frame_id,
                    rgb_b64 = Convert.ToBase64String(initialImages.Rgb),
                    depth_b64 = Convert.ToBase64String(initialImages.Depth),
                    instance_b64 = Convert.ToBase64String(initialImages.Instance),
                    wrist_rgb_b64 = Convert.ToBase64String(initialImages.WristRgb),
                },
                cancellation.Token);
            yield return Await(initialUploadTask);

            Task<ResponseEnvelope> stepTask = client.SendAsync("step", new { nworld = 1 }, cancellation.Token);
            yield return Await(stepTask);
            StateData state = stepTask.Result.state;
            visualizer.ApplyState(state);
            yield return null;

            Task<CapturePayload> captureTask = capture.CaptureAsync(state.frame_id);
            yield return Await(captureTask);
            CapturePayload images = captureTask.Result;
            Assert.That(images.Rgb.Length, Is.EqualTo(320 * 240 * 4));
            Assert.That(images.Depth.Length, Is.EqualTo(320 * 240 * 4));
            Assert.That(images.Instance.Length, Is.EqualTo(320 * 240 * 4));
            Assert.That(images.WristRgb.Length, Is.EqualTo(320 * 240 * 4));
            bool hasEncodedInstanceId = false;
            for (int index = 0; index < images.Instance.Length; index += 4)
            {
                if (images.Instance[index] != 0 || images.Instance[index + 1] != 0)
                {
                    hasEncodedInstanceId = true;
                    break;
                }
            }
            Assert.That(hasEncodedInstanceId, Is.True, "Instance image contains only background ID 0");

            Task<ResponseEnvelope> uploadTask = client.SendAsync(
                "capture",
                new
                {
                    initial = false,
                    frame_id = state.frame_id,
                    rgb_b64 = Convert.ToBase64String(images.Rgb),
                    depth_b64 = Convert.ToBase64String(images.Depth),
                    instance_b64 = Convert.ToBase64String(images.Instance),
                    wrist_rgb_b64 = Convert.ToBase64String(images.WristRgb),
                },
                cancellation.Token);
            yield return Await(uploadTask);
            Assert.That(uploadTask.Result.recorded, Is.True);

            Task<ResponseEnvelope> stopTask = client.SendAsync("record_stop", null, cancellation.Token);
            yield return Await(stopTask);
            Assert.That(stopTask.Result.frame_count, Is.EqualTo(1));
            Assert.That(File.Exists(stopTask.Result.path), Is.True, stopTask.Result.path);

            Task<ResponseEnvelope> shutdownTask = client.SendAsync("shutdown", null, cancellation.Token);
            yield return Await(shutdownTask);

            capture.Dispose();
            visualizer.Dispose();
            client.Dispose();
            cancellation.Dispose();
            UnityEngine.Object.Destroy(camera.gameObject);
        }

        private static IEnumerator Await(Task task)
        {
            while (!task.IsCompleted)
                yield return null;
            if (task.IsFaulted)
                throw task.Exception?.GetBaseException() ?? new Exception("Asynchronous test task failed");
            if (task.IsCanceled)
                throw new OperationCanceledException("Asynchronous test task was canceled");
        }

        private static IEnumerator RunPandaEpisode(
            MjWarpClient client,
            string scenario,
            string episodeId,
            CancellationToken cancellationToken)
        {
            var visualizer = new MjWarpVisualizer();
            Camera camera = null;
            Camera wristCamera = null;
            MultiModalCapture capture = null;

            Task<ResponseEnvelope> helloTask = client.SendAsync(
                "hello",
                new { client = "UnityPandaPilotTest", unity_version = Application.unityVersion, scenario },
                cancellationToken);
            yield return Await(helloTask);
            ModelSpec model = helloTask.Result.model_spec;
            Assert.That(model.scenario_id, Is.EqualTo(scenario));
            Assert.That(model.robot.id, Is.EqualTo("franka_panda"));

            camera = new GameObject($"{scenario} Front Camera").AddComponent<Camera>();
            camera.transform.position = MjWarpCoordinates.Position(model.camera_position);
            camera.transform.LookAt(MjWarpCoordinates.Position(model.camera_look_at));
            camera.fieldOfView = 48f;
            camera.nearClipPlane = 0.03f;
            camera.farClipPlane = 4f;
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = Color.black;

            visualizer.Build(model);
            capture = new MultiModalCapture(camera);
            capture.SetBindings(visualizer.RendererBindings);
            Transform hand = visualizer.FindBodyTransform(model.robot.end_effector_body);
            Assert.That(hand, Is.Not.Null);
            wristCamera = new GameObject($"{scenario} Wrist Camera").AddComponent<Camera>();
            wristCamera.transform.SetParent(hand, false);
            wristCamera.transform.localPosition = new Vector3(0f, 0.07f, 0f);
            wristCamera.transform.localRotation = Quaternion.Euler(-90f, 0f, 0f);
            wristCamera.enabled = false;
            wristCamera.fieldOfView = 62f;
            wristCamera.nearClipPlane = 0.025f;
            wristCamera.farClipPlane = 2f;
            capture.SetWristSource(wristCamera);

            Task<ResponseEnvelope> resetTask = client.SendAsync(
                "reset", new { scenario, seed = 0, policy = "expert", nworld = 1 }, cancellationToken);
            yield return Await(resetTask);
            StateData state = resetTask.Result.state;
            visualizer.ApplyState(state);

            Task<ResponseEnvelope> startTask = client.SendAsync(
                "record_start",
                new
                {
                    scenario,
                    episode_id = episodeId,
                    seed = 0,
                    policy = "expert",
                    image_width = MultiModalCapture.Width,
                    image_height = MultiModalCapture.Height,
                    unity_version = Application.unityVersion,
                    application_version = Application.version,
                    data_source = "synthetic_simulation",
                    generation_strategy = "expert_state_machine_ik",
                    license_manifest = "model/third_party/LICENSES.md",
                    camera_metadata = new
                    {
                        front = CameraMetadata(camera, "world"),
                        wrist = CameraMetadata(wristCamera, "hand"),
                        distortion_model = "none",
                        distortion_coefficients = new[] { 0f, 0f, 0f, 0f, 0f },
                    },
                },
                cancellationToken);
            yield return Await(startTask);

            Task<CapturePayload> initialCapture = capture.CaptureAsync(state.frame_id);
            yield return Await(initialCapture);
            Task<ResponseEnvelope> initialUpload = UploadCapture(
                client, initialCapture.Result, true, cancellationToken);
            yield return Await(initialUpload);

            while (!state.terminated)
            {
                Task<ResponseEnvelope> stepTask = client.SendAsync(
                    "step", new { scenario, nworld = 1 }, cancellationToken);
                yield return Await(stepTask);
                state = stepTask.Result.state;
                visualizer.ApplyState(state);
                yield return null;

                Task<CapturePayload> captureTask = capture.CaptureAsync(state.frame_id);
                yield return Await(captureTask);
                Task<ResponseEnvelope> uploadTask = UploadCapture(
                    client, captureTask.Result, false, cancellationToken);
                yield return Await(uploadTask);
            }

            Assert.That(state.success, Is.True, $"{scenario} terminated without success at frame {state.frame_id}");
            Task<ResponseEnvelope> stopTask = client.SendAsync("record_stop", null, cancellationToken);
            yield return Await(stopTask);
            Assert.That(stopTask.Result.frame_count, Is.EqualTo(state.frame_id));
            Assert.That(File.Exists(stopTask.Result.path), Is.True, stopTask.Result.path);

            capture.Dispose();
            visualizer.Dispose();
            UnityEngine.Object.Destroy(wristCamera.gameObject);
            UnityEngine.Object.Destroy(camera.gameObject);
            yield return null;
        }

        private static Task<ResponseEnvelope> UploadCapture(
            MjWarpClient client,
            CapturePayload images,
            bool initial,
            CancellationToken cancellationToken)
        {
            return client.SendAsync(
                "capture",
                new
                {
                    initial,
                    frame_id = images.FrameId,
                    rgb_b64 = Convert.ToBase64String(images.Rgb),
                    depth_b64 = Convert.ToBase64String(images.Depth),
                    instance_b64 = Convert.ToBase64String(images.Instance),
                    wrist_rgb_b64 = Convert.ToBase64String(images.WristRgb),
                },
                cancellationToken);
        }

        private static object CameraMetadata(Camera camera, string parentFrame)
        {
            float fy = 0.5f * MultiModalCapture.Height / Mathf.Tan(0.5f * camera.fieldOfView * Mathf.Deg2Rad);
            Vector3 worldPosition = camera.transform.position;
            Quaternion worldRotation = camera.transform.rotation;
            Vector3 parentPosition = parentFrame == "world" ? worldPosition : camera.transform.localPosition;
            Quaternion parentRotation = parentFrame == "world" ? worldRotation : camera.transform.localRotation;
            return new
            {
                width = MultiModalCapture.Width,
                height = MultiModalCapture.Height,
                intrinsics = new[]
                {
                    fy, 0f, MultiModalCapture.Width * 0.5f,
                    0f, fy, MultiModalCapture.Height * 0.5f,
                    0f, 0f, 1f,
                },
                position_parent_frame = new[] { parentPosition.x, parentPosition.y, parentPosition.z },
                quaternion_parent_frame_xyzw = new[]
                {
                    parentRotation.x, parentRotation.y, parentRotation.z, parentRotation.w,
                },
                position_unity_world = new[] { worldPosition.x, worldPosition.y, worldPosition.z },
                quaternion_unity_world_xyzw = new[]
                {
                    worldRotation.x, worldRotation.y, worldRotation.z, worldRotation.w,
                },
                parent_frame = parentFrame,
            };
        }
    }
}
