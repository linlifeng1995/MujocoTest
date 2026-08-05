using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace MJWarpDemo
{
    [DefaultExecutionOrder(-100)]
    public sealed class MjWarpDemoController : MonoBehaviour
    {
        private const string Host = "127.0.0.1";
        private const int Port = 8765;

        private readonly ConcurrentQueue<string> backendLogQueue = new ConcurrentQueue<string>();
        private readonly StringBuilder backendLog = new StringBuilder();
        private readonly MjWarpVisualizer visualizer = new MjWarpVisualizer();
        private readonly CancellationTokenSource lifetimeCancellation = new CancellationTokenSource();

        private MjWarpClient client;
        private MultiModalCapture capture;
        private Process backendProcess;
        private Camera mainCamera;
        private Light keyLight;
        private StateData currentState;
        private BackendInfo backendInfo;
        private BenchmarkResult[] benchmarkResults;
        private string status = "Starting";
        private string selectedPolicy = "expert";
        private string seedText = "0";
        private string lastDatasetPath = "";
        private bool recordEpisode = true;
        private bool busy;
        private bool stopRequested;
        private Vector2 logScroll;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Bootstrap()
        {
            // Batch mode is reserved for automated PlayMode integration tests, which
            // construct the bridge explicitly to avoid two clients sharing a recording.
            if (Application.isBatchMode)
                return;
            if (FindFirstObjectByType<MjWarpDemoController>() != null)
                return;
            new GameObject("MJWarp Demo Controller").AddComponent<MjWarpDemoController>();
        }

        private async void Start()
        {
            ConfigureScene();
            client = new MjWarpClient();
            try
            {
                capture = new MultiModalCapture(mainCamera);
                await ConnectBackendAsync();
            }
            catch (Exception exception)
            {
                SetError(exception);
            }
        }

        private void ConfigureScene()
        {
            mainCamera = Camera.main;
            if (mainCamera == null)
            {
                var cameraObject = new GameObject("Main Camera") { tag = "MainCamera" };
                mainCamera = cameraObject.AddComponent<Camera>();
            }
            mainCamera.transform.position = new Vector3(-0.02f, 0.86f, -0.88f);
            mainCamera.transform.LookAt(new Vector3(-0.04f, 0.035f, 0f));
            mainCamera.fieldOfView = 48f;
            mainCamera.nearClipPlane = 0.03f;
            mainCamera.farClipPlane = 4f;
            mainCamera.clearFlags = CameraClearFlags.SolidColor;
            mainCamera.backgroundColor = new Color(0.055f, 0.065f, 0.085f, 1f);

            keyLight = FindFirstObjectByType<Light>();
            if (keyLight == null)
                keyLight = new GameObject("MJWarp Key Light").AddComponent<Light>();
            keyLight.type = LightType.Directional;
            keyLight.intensity = 1.25f;
            keyLight.transform.rotation = Quaternion.Euler(48f, -32f, 0f);
            RenderSettings.ambientIntensity = 0.75f;
        }

        private async Task ConnectBackendAsync()
        {
            if (busy)
                return;
            busy = true;
            try
            {
                status = "Connecting to MJWarp backend...";
                try
                {
                    await client.ConnectAsync(Host, Port, 800, lifetimeCancellation.Token);
                }
                catch
                {
                    LaunchBackend();
                    Exception lastError = null;
                    for (int attempt = 0; attempt < 120 && !lifetimeCancellation.IsCancellationRequested; attempt++)
                    {
                        await Task.Delay(500, lifetimeCancellation.Token);
                        try
                        {
                            await client.ConnectAsync(Host, Port, 800, lifetimeCancellation.Token);
                            lastError = null;
                            break;
                        }
                        catch (Exception exception)
                        {
                            lastError = exception;
                            status = $"Waiting for CUDA backend... {attempt / 2 + 1}s";
                        }
                    }
                    if (!client.IsConnected)
                        throw new InvalidOperationException("MJWarp backend did not become ready", lastError);
                }

                ResponseEnvelope hello = await client.SendAsync("hello", new { client = "Unity", unity_version = Application.unityVersion }, lifetimeCancellation.Token);
                backendInfo = hello.backend;
                visualizer.Build(hello.model_spec);
                capture.SetBindings(visualizer.RendererBindings);
                ResponseEnvelope reset = await client.SendAsync("reset", new { seed = CurrentSeed, policy = selectedPolicy, nworld = 1 }, lifetimeCancellation.Token);
                currentState = reset.state;
                visualizer.ApplyState(currentState);
                RandomizeScene(CurrentSeed);
                status = "Ready";
            }
            finally
            {
                busy = false;
            }
        }

        private void LaunchBackend()
        {
            if (backendProcess != null && !backendProcess.HasExited)
                return;
            string projectRoot = Directory.GetParent(Application.dataPath)?.FullName
                ?? throw new InvalidOperationException("Cannot resolve Unity project root");
            string backendRoot = Path.Combine(projectRoot, "External", "MJWarpDemo");
            string python = Path.Combine(backendRoot, ".venv", "Scripts", "python.exe");
            if (!File.Exists(python))
                throw new FileNotFoundException("Project Python environment is missing. Run `uv sync --python 3.12` in External/MJWarpDemo.", python);

            var startInfo = new ProcessStartInfo
            {
                FileName = python,
                Arguments = $"-m mjwarp_demo.server --host {Host} --port {Port}",
                WorkingDirectory = backendRoot,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            };
            backendProcess = new Process { StartInfo = startInfo, EnableRaisingEvents = true };
            backendProcess.OutputDataReceived += (_, args) => EnqueueBackendLog(args.Data);
            backendProcess.ErrorDataReceived += (_, args) => EnqueueBackendLog(args.Data);
            backendProcess.Exited += (_, __) => EnqueueBackendLog($"Backend exited with code {backendProcess.ExitCode}");
            if (!backendProcess.Start())
                throw new InvalidOperationException("Failed to start MJWarp backend process");
            backendProcess.BeginOutputReadLine();
            backendProcess.BeginErrorReadLine();
        }

        private void EnqueueBackendLog(string line)
        {
            if (!string.IsNullOrWhiteSpace(line))
                backendLogQueue.Enqueue(line);
        }

        private void Update()
        {
            while (backendLogQueue.TryDequeue(out string line))
            {
                backendLog.AppendLine(line);
                if (backendLog.Length > 12000)
                    backendLog.Remove(0, backendLog.Length - 9000);
            }
        }

        private async void RunSingleEpisode()
        {
            await RunOperation(async () => await RunEpisodeAsync(selectedPolicy, CurrentSeed, recordEpisode));
        }

        private async void GenerateAcceptanceSet()
        {
            await RunOperation(async () =>
            {
                int baseSeed = CurrentSeed;
                for (int index = 0; index < 10 && !stopRequested; index++)
                    await RunEpisodeAsync("expert", baseSeed + index, true);
                for (int index = 0; index < 10 && !stopRequested; index++)
                    await RunEpisodeAsync("random", baseSeed + 100 + index, true);
            });
        }

        private async Task RunEpisodeAsync(string policy, int seed, bool record)
        {
            if (!client.IsConnected)
                await ConnectBackendAsync();
            status = $"Resetting {policy} episode (seed {seed})";
            ResponseEnvelope reset = await client.SendAsync("reset", new { seed, policy, nworld = 1 }, lifetimeCancellation.Token);
            currentState = reset.state;
            visualizer.ApplyState(currentState);
            visualizer.RandomizeAppearance(seed);
            RandomizeScene(seed);

            bool recordingStarted = false;
            if (record)
            {
                ResponseEnvelope started = await client.SendAsync(
                    "record_start",
                    new { seed, policy, image_width = MultiModalCapture.Width, image_height = MultiModalCapture.Height },
                    lifetimeCancellation.Token);
                recordingStarted = true;
                status = $"Recording {started.episode_id}";
            }

            try
            {
                while (!stopRequested && !currentState.terminated)
                {
                    ResponseEnvelope stepped = await client.SendAsync("step", new { nworld = 1 }, lifetimeCancellation.Token);
                    currentState = stepped.state;
                    visualizer.ApplyState(currentState);
                    await Task.Yield();

                    CapturePayload images = await capture.CaptureAsync(currentState.frame_id);
                    if (record)
                    {
                        await client.SendAsync(
                            "capture",
                            new
                            {
                                frame_id = images.FrameId,
                                rgb_b64 = Convert.ToBase64String(images.Rgb),
                                depth_b64 = Convert.ToBase64String(images.Depth),
                                instance_b64 = Convert.ToBase64String(images.Instance),
                            },
                            lifetimeCancellation.Token);
                    }
                    status = $"{policy} | frame {currentState.frame_id}/120 | reward {currentState.reward:F3}";
                }
            }
            finally
            {
                if (recordingStarted && client.IsConnected)
                {
                    ResponseEnvelope stopped = await client.SendAsync("record_stop", null, lifetimeCancellation.Token);
                    lastDatasetPath = stopped.path;
                }
            }
            status = $"Episode complete: success={currentState.success}, frames={currentState.frame_id}";
        }

        private async void RunBenchmark()
        {
            await RunOperation(async () =>
            {
                status = "Benchmarking 1/64/256/1024 worlds...";
                ResponseEnvelope result = await client.SendAsync(
                    "benchmark",
                    new { sizes = new[] { 1, 64, 256, 1024 }, steps = 300, warmup = 30 },
                    lifetimeCancellation.Token);
                benchmarkResults = result.results;
                status = "Benchmark complete";
            });
        }

        private async Task RunOperation(Func<Task> operation)
        {
            if (busy)
                return;
            busy = true;
            stopRequested = false;
            try
            {
                await operation();
            }
            catch (OperationCanceledException) when (lifetimeCancellation.IsCancellationRequested)
            {
                status = "Stopped";
            }
            catch (Exception exception)
            {
                SetError(exception);
            }
            finally
            {
                busy = false;
            }
        }

        private void RandomizeScene(int seed)
        {
            var random = new System.Random(seed);
            keyLight.intensity = 0.95f + (float)random.NextDouble() * 0.55f;
            keyLight.transform.rotation = Quaternion.Euler(
                40f + (float)random.NextDouble() * 18f,
                -45f + (float)random.NextDouble() * 30f,
                0f);
            float cameraX = -0.02f + ((float)random.NextDouble() - 0.5f) * 0.05f;
            float cameraZ = -0.88f + ((float)random.NextDouble() - 0.5f) * 0.05f;
            mainCamera.transform.position = new Vector3(cameraX, 0.86f, cameraZ);
            mainCamera.transform.LookAt(new Vector3(-0.04f, 0.035f, 0f));
        }

        private int CurrentSeed => int.TryParse(seedText, out int seed) ? seed : 0;

        private void SetError(Exception exception)
        {
            status = $"ERROR: {exception.GetBaseException().Message}";
            Debug.LogException(exception);
        }

        private void OnGUI()
        {
            const float panelWidth = 390f;
            GUILayout.BeginArea(new Rect(12f, 12f, panelWidth, Screen.height - 24f), GUI.skin.box);
            GUILayout.Label("MJWarp × Unity Embodied Data Demo", HeaderStyle());
            GUILayout.Label($"Status: {status}");
            GUILayout.Label(backendInfo == null
                ? "Backend: not connected"
                : $"Backend: MuJoCo/MJWarp {backendInfo.mujoco_warp_version} | {backendInfo.gpu}");

            GUILayout.Space(6f);
            GUILayout.BeginHorizontal();
            GUILayout.Label("Seed", GUILayout.Width(42f));
            seedText = GUILayout.TextField(seedText, GUILayout.Width(90f));
            GUI.enabled = !busy;
            if (GUILayout.Button("Expert")) selectedPolicy = "expert";
            if (GUILayout.Button("Random")) selectedPolicy = "random";
            GUI.enabled = true;
            GUILayout.EndHorizontal();
            GUILayout.Label($"Policy: {selectedPolicy}");
            recordEpisode = GUILayout.Toggle(recordEpisode, "Record aligned HDF5 episode");

            GUILayout.BeginHorizontal();
            GUI.enabled = !busy && client != null && client.IsConnected;
            if (GUILayout.Button("Run Episode")) RunSingleEpisode();
            if (GUILayout.Button("Generate 20")) GenerateAcceptanceSet();
            GUI.enabled = busy;
            if (GUILayout.Button("Stop")) stopRequested = true;
            GUI.enabled = true;
            GUILayout.EndHorizontal();

            GUILayout.BeginHorizontal();
            GUI.enabled = !busy;
            if (GUILayout.Button("Connect / Retry")) _ = ConnectBackendAsync();
            GUI.enabled = !busy && client != null && client.IsConnected;
            if (GUILayout.Button("GPU Benchmark")) RunBenchmark();
            GUI.enabled = true;
            GUILayout.EndHorizontal();

            if (currentState != null)
            {
                GUILayout.Space(6f);
                GUILayout.Label($"Frame {currentState.frame_id} | sim {currentState.sim_time:F3}s");
                GUILayout.Label($"Reward {currentState.reward:F4} | contacts {currentState.contacts?.count ?? 0}");
                GUILayout.Label($"Success {currentState.success} | terminated {currentState.terminated}");
                if (currentState.metrics != null)
                    GUILayout.Label($"Interactive physics: {currentState.metrics.physics_steps_per_second:F0} steps/s");
            }

            if (!string.IsNullOrEmpty(lastDatasetPath))
                GUILayout.Label($"Last HDF5: {lastDatasetPath}");

            if (benchmarkResults != null)
            {
                GUILayout.Space(6f);
                GUILayout.Label("CUDA graph benchmark");
                foreach (BenchmarkResult item in benchmarkResults)
                {
                    string value = string.IsNullOrEmpty(item.error)
                        ? $"{item.actual_nworld,4} worlds: {item.physics_steps_per_second,12:N0} steps/s{(item.fallback ? " (fallback)" : "") }"
                        : $"{item.requested_nworld,4} worlds: ERROR {item.error}";
                    GUILayout.Label(value);
                }
            }

            if (capture != null)
            {
                GUILayout.Space(6f);
                GUILayout.Label("RGB / Linear depth / Instance ID");
                float previewWidth = (panelWidth - 34f) / 3f;
                Rect row = GUILayoutUtility.GetRect(panelWidth - 20f, previewWidth * 0.75f);
                GUI.DrawTexture(new Rect(row.x, row.y, previewWidth, row.height), capture.RgbTexture, ScaleMode.ScaleToFit, false);
                GUI.DrawTexture(new Rect(row.x + previewWidth + 5f, row.y, previewWidth, row.height), capture.DepthTexture, ScaleMode.ScaleToFit, false);
                GUI.DrawTexture(new Rect(row.x + (previewWidth + 5f) * 2f, row.y, previewWidth, row.height), capture.InstanceTexture, ScaleMode.ScaleToFit, false);
            }

            GUILayout.Space(6f);
            GUILayout.Label("Backend log", GUILayout.ExpandWidth(true));
            logScroll = GUILayout.BeginScrollView(logScroll, GUILayout.Height(120f));
            GUILayout.Label(backendLog.ToString());
            GUILayout.EndScrollView();
            GUILayout.EndArea();
        }

        private static GUIStyle HeaderStyle()
        {
            var style = new GUIStyle(GUI.skin.label)
            {
                fontSize = 18,
                fontStyle = FontStyle.Bold,
            };
            return style;
        }

        private void OnDestroy()
        {
            lifetimeCancellation.Cancel();
            capture?.Dispose();
            visualizer.Dispose();
            client?.Dispose();
            if (backendProcess != null)
            {
                try
                {
                    if (!backendProcess.HasExited)
                        backendProcess.Kill();
                }
                catch (Exception)
                {
                    // Editor shutdown: the OS will reclaim the child process if it already exited.
                }
                backendProcess.Dispose();
            }
            lifetimeCancellation.Dispose();
        }
    }
}
