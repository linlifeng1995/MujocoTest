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
        private Font chineseFont;
        private StateData currentState;
        private BackendInfo backendInfo;
        private BenchmarkResult[] benchmarkResults;
        private string status = "正在启动";
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

            string[] preferredFonts = { "Microsoft YaHei UI", "Microsoft YaHei", "SimHei" };
            string[] installedFonts = Font.GetOSInstalledFontNames();
            string selectedFont = preferredFonts.FirstOrDefault(candidate =>
                installedFonts.Any(installed => string.Equals(installed, candidate, StringComparison.OrdinalIgnoreCase)));
            if (!string.IsNullOrEmpty(selectedFont))
                chineseFont = Font.CreateDynamicFontFromOSFont(selectedFont, 16);
        }

        private async Task ConnectBackendAsync()
        {
            if (busy)
                return;
            busy = true;
            try
            {
                status = "正在连接 MJWarp 后端……";
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
                            status = $"正在等待 CUDA 后端……{attempt / 2 + 1} 秒";
                        }
                    }
                    if (!client.IsConnected)
                        throw new InvalidOperationException("MJWarp 后端未能就绪", lastError);
                }

                ResponseEnvelope hello = await client.SendAsync("hello", new { client = "Unity", unity_version = Application.unityVersion }, lifetimeCancellation.Token);
                backendInfo = hello.backend;
                visualizer.Build(hello.model_spec);
                capture.SetBindings(visualizer.RendererBindings);
                ResponseEnvelope reset = await client.SendAsync("reset", new { seed = CurrentSeed, policy = selectedPolicy, nworld = 1 }, lifetimeCancellation.Token);
                currentState = reset.state;
                visualizer.ApplyState(currentState);
                RandomizeScene(CurrentSeed);
                status = "就绪";
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
                ?? throw new InvalidOperationException("无法确定 Unity 工程根目录");
            string backendRoot = Path.Combine(projectRoot, "External", "MJWarpDemo");
            string python = Path.Combine(backendRoot, ".venv", "Scripts", "python.exe");
            if (!File.Exists(python))
                throw new FileNotFoundException("缺少项目 Python 环境。请在 External/MJWarpDemo 中运行 `uv sync --python 3.12`。", python);

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
            backendProcess.Exited += (_, __) => EnqueueBackendLog($"后端进程已退出，退出码：{backendProcess.ExitCode}");
            if (!backendProcess.Start())
                throw new InvalidOperationException("启动 MJWarp 后端进程失败");
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
            status = $"正在重置{PolicyDisplayName(policy)}回合（随机种子 {seed}）";
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
                status = $"正在录制：{started.episode_id}";
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
                    status = $"{PolicyDisplayName(policy)}｜帧 {currentState.frame_id}/120｜奖励 {currentState.reward:F3}";
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
            status = $"回合完成：成功={YesNo(currentState.success)}，帧数={currentState.frame_id}";
        }

        private async void RunBenchmark()
        {
            await RunOperation(async () =>
            {
                status = "正在测试 1/64/256/1024 个并行环境……";
                ResponseEnvelope result = await client.SendAsync(
                    "benchmark",
                    new { sizes = new[] { 1, 64, 256, 1024 }, steps = 300, warmup = 30 },
                    lifetimeCancellation.Token);
                benchmarkResults = result.results;
                status = "GPU 性能测试完成";
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
                status = "已停止";
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
            status = $"错误：{exception.GetBaseException().Message}";
            Debug.LogException(exception);
        }

        private void OnGUI()
        {
            if (chineseFont != null)
                GUI.skin.font = chineseFont;
            const float panelWidth = 390f;
            GUILayout.BeginArea(new Rect(12f, 12f, panelWidth, Screen.height - 24f), GUI.skin.box);
            GUILayout.Label("MJWarp × Unity 具身训练数据演示", HeaderStyle());
            GUILayout.Label($"状态：{status}");
            GUILayout.Label(backendInfo == null
                ? "后端：未连接"
                : $"后端：MuJoCo/MJWarp {backendInfo.mujoco_warp_version}｜{backendInfo.gpu}");

            GUILayout.Space(6f);
            GUILayout.BeginHorizontal();
            GUILayout.Label("随机种子", GUILayout.Width(68f));
            seedText = GUILayout.TextField(seedText, GUILayout.Width(90f));
            GUI.enabled = !busy;
            if (GUILayout.Button("专家策略")) selectedPolicy = "expert";
            if (GUILayout.Button("随机策略")) selectedPolicy = "random";
            GUI.enabled = true;
            GUILayout.EndHorizontal();
            GUILayout.Label($"当前策略：{PolicyDisplayName(selectedPolicy)}");
            recordEpisode = GUILayout.Toggle(recordEpisode, "录制严格对齐的 HDF5 回合数据");

            GUILayout.BeginHorizontal();
            GUI.enabled = !busy && client != null && client.IsConnected;
            if (GUILayout.Button("运行单回合")) RunSingleEpisode();
            if (GUILayout.Button("生成 20 回合")) GenerateAcceptanceSet();
            GUI.enabled = busy;
            if (GUILayout.Button("停止")) stopRequested = true;
            GUI.enabled = true;
            GUILayout.EndHorizontal();

            GUILayout.BeginHorizontal();
            GUI.enabled = !busy;
            if (GUILayout.Button("连接 / 重试")) _ = ConnectBackendAsync();
            GUI.enabled = !busy && client != null && client.IsConnected;
            if (GUILayout.Button("GPU 性能测试")) RunBenchmark();
            GUI.enabled = true;
            GUILayout.EndHorizontal();

            if (currentState != null)
            {
                GUILayout.Space(6f);
                GUILayout.Label($"帧 {currentState.frame_id}｜仿真时间 {currentState.sim_time:F3} 秒");
                GUILayout.Label($"奖励 {currentState.reward:F4}｜接触数 {currentState.contacts?.count ?? 0}");
                GUILayout.Label($"成功 {YesNo(currentState.success)}｜已结束 {YesNo(currentState.terminated)}");
                if (currentState.metrics != null)
                    GUILayout.Label($"交互物理吞吐：{currentState.metrics.physics_steps_per_second:F0} 步/秒");
            }

            if (!string.IsNullOrEmpty(lastDatasetPath))
                GUILayout.Label($"最近生成的 HDF5：{lastDatasetPath}");

            if (benchmarkResults != null)
            {
                GUILayout.Space(6f);
                GUILayout.Label("CUDA Graph 性能测试");
                foreach (BenchmarkResult item in benchmarkResults)
                {
                    string value = string.IsNullOrEmpty(item.error)
                        ? $"{item.actual_nworld,4} 个环境：{item.physics_steps_per_second,12:N0} 步/秒{(item.fallback ? "（已降级）" : "") }"
                        : $"{item.requested_nworld,4} 个环境：错误 {item.error}";
                    GUILayout.Label(value);
                }
            }

            if (capture != null)
            {
                GUILayout.Space(6f);
                GUILayout.Label("RGB / 线性深度 / 实例 ID");
                float previewWidth = (panelWidth - 34f) / 3f;
                Rect row = GUILayoutUtility.GetRect(panelWidth - 20f, previewWidth * 0.75f);
                GUI.DrawTexture(new Rect(row.x, row.y, previewWidth, row.height), capture.RgbTexture, ScaleMode.ScaleToFit, false);
                GUI.DrawTexture(new Rect(row.x + previewWidth + 5f, row.y, previewWidth, row.height), capture.DepthTexture, ScaleMode.ScaleToFit, false);
                GUI.DrawTexture(new Rect(row.x + (previewWidth + 5f) * 2f, row.y, previewWidth, row.height), capture.InstanceTexture, ScaleMode.ScaleToFit, false);
            }

            GUILayout.Space(6f);
            GUILayout.Label("后端日志", GUILayout.ExpandWidth(true));
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

        private static string PolicyDisplayName(string policy)
        {
            return policy == "expert" ? "专家策略" : "随机策略";
        }

        private static string YesNo(bool value)
        {
            return value ? "是" : "否";
        }

        private void OnDestroy()
        {
            lifetimeCancellation.Cancel();
            if (chineseFont != null)
                Destroy(chineseFont);
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
