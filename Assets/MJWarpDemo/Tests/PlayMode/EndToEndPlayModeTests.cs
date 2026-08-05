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
                    frame_id = state.frame_id,
                    rgb_b64 = Convert.ToBase64String(images.Rgb),
                    depth_b64 = Convert.ToBase64String(images.Depth),
                    instance_b64 = Convert.ToBase64String(images.Instance),
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
    }
}
