using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.SceneManagement;
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
        private Camera wristCamera;
        private Light keyLight;
        private Font chineseFont;
        private MjWarpScenarioInfo scenario;
        private ModelSpec modelSpec;
        private StateData currentState;
        private BackendInfo backendInfo;
        private BenchmarkResult[] benchmarkResults;
        private ModelArtifactInfo[] availableModels = Array.Empty<ModelArtifactInfo>();
        private LoadedModelInfo loadedModel;
        private InferenceInfo lastInference;
        private string status = "正在启动";
        private string selectedPolicy = "expert";
        private string seedText = "0";
        private string batchCountText = "20";
        private string batchSummary = "";
        private string lastDatasetPath = "";
        private float episodeTotalReward;
        private int selectedModelIndex;
        private bool recordEpisode = true;
        private bool busy;
        private bool stopRequested;
        private Vector2 panelScroll;
        private Vector2 logScroll;
        private Vector3 baseCameraPosition = new Vector3(-0.02f, 0.86f, -0.88f);
        private Vector3 baseCameraLookAt = new Vector3(-0.04f, 0.035f, 0f);

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
            scenario = MjWarpScenarioCatalog.Active;
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
            mainCamera.transform.position = baseCameraPosition;
            mainCamera.transform.LookAt(baseCameraLookAt);
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

                ResponseEnvelope hello = await client.SendAsync(
                    "hello",
                    new { client = "Unity", unity_version = Application.unityVersion, scenario = scenario.ScenarioId },
                    lifetimeCancellation.Token);
                backendInfo = hello.backend;
                modelSpec = hello.model_spec;
                availableModels = hello.models ?? Array.Empty<ModelArtifactInfo>();
                selectedModelIndex = Mathf.Clamp(selectedModelIndex, 0, Math.Max(0, availableModels.Length - 1));
                scenario = MjWarpScenarioCatalog.FindById(modelSpec.scenario_id);
                ApplyModelPresentation(modelSpec);
                visualizer.Build(modelSpec);
                capture.SetBindings(visualizer.RendererBindings);
                ConfigureWristCamera(modelSpec);
                ResponseEnvelope reset = await client.SendAsync(
                    "reset",
                    new { seed = CurrentSeed, policy = selectedPolicy, nworld = 1, scenario = scenario.ScenarioId },
                    lifetimeCancellation.Token);
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
                Arguments = $"-m mjwarp_demo.server --host {Host} --port {Port} --scenario {scenario.ScenarioId}",
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
            await RunOperation(async () => { await RunEpisodeAsync(selectedPolicy, CurrentSeed, recordEpisode); });
        }

        private async void GenerateBatchSet()
        {
            await RunOperation(async () =>
            {
                int baseSeed = CurrentSeed;
                int count = CurrentBatchCount;
                int completed = 0;
                int succeeded = 0;
                for (int index = 0; index < count && !stopRequested; index++)
                {
                    if (await RunEpisodeAsync(selectedPolicy, baseSeed + index, recordEpisode))
                        succeeded++;
                    completed++;
                    batchSummary = $"批量进度 {completed}/{count}｜成功 {succeeded}｜失败 {completed - succeeded}";
                }
                batchSummary = $"批量完成 {completed}/{count}｜成功 {succeeded}｜失败 {completed - succeeded}";
            });
        }

        private async void GenerateStandardDataset()
        {
            await RunOperation(async () =>
            {
                int baseSeed = CurrentSeed;
                int completed = 0;
                int succeeded = 0;
                foreach (var batch in new[]
                {
                    new { Policy = "expert", Count = 375, Offset = 0 },
                    new { Policy = "perturbed", Count = 125, Offset = 100000 },
                })
                {
                    for (int index = 0; index < batch.Count && !stopRequested; index++)
                    {
                        if (await RunEpisodeAsync(batch.Policy, baseSeed + batch.Offset + index, true))
                            succeeded++;
                        completed++;
                        batchSummary = $"当前任务 Pilot {completed}/500｜成功 {succeeded}｜失败 {completed - succeeded}";
                    }
                }
                batchSummary = $"当前任务 Pilot 完成 {completed}/500｜成功 {succeeded}｜失败 {completed - succeeded}";
            });
        }

        private async Task<bool> RunEpisodeAsync(string policy, int seed, bool record)
        {
            if (!client.IsConnected)
                await ConnectBackendAsync();
            status = $"正在重置{PolicyDisplayName(policy)}回合（随机种子 {seed}）";
            ResponseEnvelope reset = await client.SendAsync(
                "reset",
                new { seed, policy, nworld = 1, scenario = scenario.ScenarioId },
                lifetimeCancellation.Token);
            currentState = reset.state;
            visualizer.ApplyState(currentState);
            visualizer.RandomizeAppearance(seed);
            RandomizeScene(seed);
            episodeTotalReward = 0f;
            lastInference = null;

            bool recordingStarted = false;
            if (record)
            {
                ResponseEnvelope started = await client.SendAsync(
                    "record_start",
                    new
                    {
                        seed,
                        policy,
                        scenario = scenario.ScenarioId,
                        image_width = MultiModalCapture.Width,
                        image_height = MultiModalCapture.Height,
                        unity_version = Application.unityVersion,
                        application_version = Application.version,
                        code_version = "mini-pilot-v0.2",
                        schema_profile = "panda-mini-pilot-v0.2",
                        data_source = "synthetic_simulation",
                        generation_strategy = policy,
                        license_manifest = "model/third_party/LICENSES.md",
                        camera_metadata = new
                        {
                            front = CameraMetadata(mainCamera, "world"),
                            wrist = CameraMetadata(wristCamera != null ? wristCamera : mainCamera, "hand"),
                            distortion_model = "none",
                            distortion_coefficients = new[] { 0f, 0f, 0f, 0f, 0f },
                        },
                    },
                    lifetimeCancellation.Token);
                recordingStarted = true;
                status = $"正在录制：{started.episode_id}";

                CapturePayload initialImages = await capture.CaptureAsync(currentState.frame_id);
                await client.SendAsync(
                    "capture",
                    new
                    {
                        initial = true,
                        frame_id = initialImages.FrameId,
                        rgb_b64 = Convert.ToBase64String(initialImages.Rgb),
                        depth_b64 = Convert.ToBase64String(initialImages.Depth),
                        instance_b64 = Convert.ToBase64String(initialImages.Instance),
                        wrist_rgb_b64 = Convert.ToBase64String(initialImages.WristRgb),
                        wrist_instance_b64 = Convert.ToBase64String(initialImages.WristInstance),
                    },
                    lifetimeCancellation.Token);
            }

            try
            {
                while (!stopRequested && !currentState.terminated)
                {
                    ResponseEnvelope stepped = await client.SendAsync(
                        "step",
                        new { nworld = 1, scenario = scenario.ScenarioId },
                        lifetimeCancellation.Token);
                    currentState = stepped.state;
                    lastInference = stepped.inference;
                    episodeTotalReward += currentState.reward;
                    visualizer.ApplyState(currentState);
                    await Task.Yield();

                    CapturePayload images = await capture.CaptureAsync(currentState.frame_id);
                    if (record)
                    {
                        await client.SendAsync(
                            "capture",
                            new
                            {
                                initial = false,
                                frame_id = images.FrameId,
                                rgb_b64 = Convert.ToBase64String(images.Rgb),
                                depth_b64 = Convert.ToBase64String(images.Depth),
                                instance_b64 = Convert.ToBase64String(images.Instance),
                                wrist_rgb_b64 = Convert.ToBase64String(images.WristRgb),
                                wrist_instance_b64 = Convert.ToBase64String(images.WristInstance),
                            },
                            lifetimeCancellation.Token);
                    }
                    status = $"{PolicyDisplayName(policy)}｜帧 {currentState.frame_id}/{modelSpec?.max_frames ?? 120}｜奖励 {currentState.reward:F3}";
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
            status = $"回合完成：成功={YesNo(currentState.success)}，帧数={currentState.frame_id}，累计奖励={episodeTotalReward:F3}";
            return currentState.success;
        }

        private async void RefreshModels()
        {
            await RunOperation(async () =>
            {
                ResponseEnvelope response = await client.SendAsync(
                    "model_list",
                    new { scenario = scenario.ScenarioId },
                    lifetimeCancellation.Token);
                availableModels = response.models ?? Array.Empty<ModelArtifactInfo>();
                selectedModelIndex = Mathf.Clamp(selectedModelIndex, 0, Math.Max(0, availableModels.Length - 1));
                status = availableModels.Length == 0 ? "当前场景没有可用的行为克隆模型" : $"发现 {availableModels.Length} 个学习策略模型";
            });
        }

        private async void LoadSelectedModel()
        {
            if (availableModels.Length == 0)
                return;
            await RunOperation(async () =>
            {
                ModelArtifactInfo selected = availableModels[selectedModelIndex];
                status = $"正在加载模型：{selected.artifact_id}";
                ResponseEnvelope response = await client.SendAsync(
                    "model_load",
                    new { scenario = scenario.ScenarioId, artifact_id = selected.artifact_id },
                    lifetimeCancellation.Token);
                loadedModel = response.model_info;
                selectedPolicy = "learned";
                status = $"学习策略已加载：{response.loaded_model}";
            });
        }

        private async void UnloadModel()
        {
            await RunOperation(async () =>
            {
                await client.SendAsync("model_unload", null, lifetimeCancellation.Token);
                loadedModel = null;
                lastInference = null;
                if (selectedPolicy == "learned")
                    selectedPolicy = "expert";
                status = "学习策略已卸载";
            });
        }

        private async void RunBenchmark()
        {
            await RunOperation(async () =>
            {
                status = "正在测试 1/64/256/1024 个并行环境……";
                ResponseEnvelope result = await client.SendAsync(
                    "benchmark",
                    new
                    {
                        scenario = scenario.ScenarioId,
                        sizes = new[] { 1, 64, 256, 1024 },
                        steps = 300,
                        warmup = 30,
                    },
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
            float cameraX = baseCameraPosition.x + ((float)random.NextDouble() - 0.5f) * 0.05f;
            float cameraZ = baseCameraPosition.z + ((float)random.NextDouble() - 0.5f) * 0.05f;
            mainCamera.transform.position = new Vector3(cameraX, baseCameraPosition.y, cameraZ);
            mainCamera.transform.LookAt(baseCameraLookAt);
        }

        private void ApplyModelPresentation(ModelSpec spec)
        {
            if (spec?.camera_position != null && spec.camera_position.Length >= 3)
                baseCameraPosition = MjWarpCoordinates.Position(spec.camera_position);
            if (spec?.camera_look_at != null && spec.camera_look_at.Length >= 3)
                baseCameraLookAt = MjWarpCoordinates.Position(spec.camera_look_at);
            if (spec != null && spec.camera_fov_degrees > 0f)
                mainCamera.fieldOfView = spec.camera_fov_degrees;
            if (spec != null && spec.camera_near_clip_m > 0f)
                mainCamera.nearClipPlane = spec.camera_near_clip_m;
            mainCamera.transform.position = baseCameraPosition;
            mainCamera.transform.LookAt(baseCameraLookAt);
        }

        private void ConfigureWristCamera(ModelSpec spec)
        {
            if (wristCamera != null)
                Destroy(wristCamera.gameObject);
            wristCamera = null;
            if (spec?.robot == null || !string.Equals(spec.robot.id, "franka_panda", StringComparison.Ordinal))
            {
                capture?.SetWristSource(null);
                return;
            }
            Transform hand = visualizer.FindBodyTransform(spec.robot.end_effector_body);
            if (hand == null)
                return;
            var cameraObject = new GameObject("Panda Wrist Camera");
            cameraObject.transform.SetParent(hand, false);
            // Mount the camera beside the gripper instead of on the hand centreline.
            // A centreline camera is occluded by link7 behind the hand, while the old
            // forward mount moved past the object as soon as the fingers closed.
            cameraObject.transform.localPosition = new Vector3(0.07f, 0.08f, 0f);
            cameraObject.transform.localRotation = Quaternion.LookRotation(
                new Vector3(-0.07f, -0.025f, 0f).normalized,
                Vector3.forward);
            wristCamera = cameraObject.AddComponent<Camera>();
            wristCamera.enabled = false;
            wristCamera.fieldOfView = 96f;
            wristCamera.nearClipPlane = 0.015f;
            wristCamera.farClipPlane = 2.0f;
            capture?.SetWristSource(wristCamera);
        }

        private static object CameraMetadata(Camera camera, string parentFrame)
        {
            float fy = 0.5f * MultiModalCapture.Height / Mathf.Tan(0.5f * camera.fieldOfView * Mathf.Deg2Rad);
            float fx = fy;
            Vector3 worldPosition = camera.transform.position;
            Quaternion worldRotation = camera.transform.rotation;
            Vector3 parentPosition = parentFrame == "world"
                ? worldPosition
                : camera.transform.localPosition;
            Quaternion parentRotation = parentFrame == "world"
                ? worldRotation
                : camera.transform.localRotation;
            return new
            {
                width = MultiModalCapture.Width,
                height = MultiModalCapture.Height,
                intrinsics = new[] { fx, 0f, MultiModalCapture.Width * 0.5f, 0f, fy, MultiModalCapture.Height * 0.5f, 0f, 0f, 1f },
                position_parent_frame = new[] { parentPosition.x, parentPosition.y, parentPosition.z },
                quaternion_parent_frame_xyzw = new[] { parentRotation.x, parentRotation.y, parentRotation.z, parentRotation.w },
                position_unity_world = new[] { worldPosition.x, worldPosition.y, worldPosition.z },
                quaternion_unity_world_xyzw = new[] { worldRotation.x, worldRotation.y, worldRotation.z, worldRotation.w },
                parent_frame = parentFrame,
                vertical_fov_degrees = camera.fieldOfView,
                near_clip_m = camera.nearClipPlane,
                far_clip_m = camera.farClipPlane,
            };
        }

        private int CurrentSeed => int.TryParse(seedText, out int seed) ? seed : 0;

        private int CurrentBatchCount => int.TryParse(batchCountText, out int count)
            ? Mathf.Clamp(count, 1, 1000)
            : 20;

        private void SetError(Exception exception)
        {
            status = $"错误：{exception.GetBaseException().Message}";
            Debug.LogException(exception);
        }

        private void OnGUI()
        {
            if (chineseFont != null)
                GUI.skin.font = chineseFont;
            const float panelWidth = 440f;
            GUILayout.BeginArea(new Rect(12f, 12f, panelWidth, Screen.height - 24f), GUI.skin.box);
            panelScroll = GUILayout.BeginScrollView(panelScroll);
            GUILayout.Label($"MJWarp × Unity｜{scenario?.DisplayName ?? "具身训练数据演示"}", HeaderStyle());
            if (scenario != null)
            {
                GUILayout.Label($"业务类型：{scenario.BusinessType}");
                GUILayout.Label(scenario.Description, GUI.skin.label);
            }
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
            if (GUILayout.Button("扰动策略")) selectedPolicy = "perturbed";
            if (GUILayout.Button("随机策略")) selectedPolicy = "random";
            GUI.enabled = !busy && loadedModel != null;
            if (GUILayout.Button("学习策略")) selectedPolicy = "learned";
            GUI.enabled = true;
            GUILayout.EndHorizontal();
            GUILayout.Label($"当前策略：{PolicyDisplayName(selectedPolicy)}");
            recordEpisode = GUILayout.Toggle(recordEpisode, "录制严格对齐的 HDF5 回合数据");

            GUILayout.Space(4f);
            GUILayout.Label("学习策略模型");
            if (availableModels.Length == 0)
            {
                GUILayout.Label("当前场景尚无行为克隆模型，请先运行训练命令。");
            }
            else
            {
                ModelArtifactInfo selectedModel = availableModels[selectedModelIndex];
                GUILayout.Label($"候选 {selectedModelIndex + 1}/{availableModels.Length}：{selectedModel.artifact_id}");
                GUILayout.Label($"验证 MSE：{selectedModel.validation_mse:F6}");
                GUILayout.BeginHorizontal();
                GUI.enabled = !busy && availableModels.Length > 1;
                if (GUILayout.Button("上一个")) selectedModelIndex = (selectedModelIndex - 1 + availableModels.Length) % availableModels.Length;
                if (GUILayout.Button("下一个")) selectedModelIndex = (selectedModelIndex + 1) % availableModels.Length;
                GUI.enabled = !busy && client != null && client.IsConnected;
                if (GUILayout.Button("加载模型")) LoadSelectedModel();
                GUILayout.EndHorizontal();
            }
            GUILayout.BeginHorizontal();
            GUI.enabled = !busy && client != null && client.IsConnected;
            if (GUILayout.Button("刷新模型列表")) RefreshModels();
            GUI.enabled = !busy && loadedModel != null;
            if (GUILayout.Button("卸载模型")) UnloadModel();
            GUI.enabled = true;
            GUILayout.EndHorizontal();
            GUILayout.Label(loadedModel == null
                ? "已加载模型：无"
                : $"已加载：{loadedModel.artifact_id}｜设备 {loadedModel.device}");

            GUILayout.Space(4f);
            GUILayout.Label("切换业务场景");
            GUI.enabled = !busy;
            for (int rowIndex = 0; rowIndex < MjWarpScenarioCatalog.All.Count; rowIndex += 2)
            {
                GUILayout.BeginHorizontal();
                for (int column = 0; column < 2 && rowIndex + column < MjWarpScenarioCatalog.All.Count; column++)
                {
                    MjWarpScenarioInfo item = MjWarpScenarioCatalog.All[rowIndex + column];
                    string label = item.ScenarioId == scenario?.ScenarioId ? $"● {item.DisplayName}" : item.DisplayName;
                    if (GUILayout.Button(label))
                        SceneManager.LoadScene(item.SceneName);
                }
                GUILayout.EndHorizontal();
            }
            GUI.enabled = true;

            GUILayout.BeginHorizontal();
            GUI.enabled = !busy && client != null && client.IsConnected;
            if (GUILayout.Button("运行单回合")) RunSingleEpisode();
            GUILayout.Label("批量数", GUILayout.Width(48f));
            batchCountText = GUILayout.TextField(batchCountText, GUILayout.Width(48f));
            if (GUILayout.Button("批量采集")) GenerateBatchSet();
            GUI.enabled = busy;
            if (GUILayout.Button("停止")) stopRequested = true;
            GUI.enabled = true;
            GUILayout.EndHorizontal();

            GUILayout.BeginHorizontal();
            GUI.enabled = !busy && client != null && client.IsConnected;
            if (GUILayout.Button("生成当前任务 500 回合（专家375 + 扰动125）")) GenerateStandardDataset();
            GUI.enabled = true;
            GUILayout.EndHorizontal();
            if (!string.IsNullOrEmpty(batchSummary))
                GUILayout.Label(batchSummary);

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
                GUILayout.Label($"累计奖励 {episodeTotalReward:F4}");
                GUILayout.Label($"任务阶段 {currentState.task_stage}｜距目标 {currentState.distance_to_goal:F3} 米");
                GUILayout.Label($"成功 {YesNo(currentState.success)}｜已结束 {YesNo(currentState.terminated)}");
                if (currentState.metrics != null)
                    GUILayout.Label($"交互物理吞吐：{currentState.metrics.physics_steps_per_second:F0} 步/秒");
            }

            if (lastInference != null)
            {
                string action = lastInference.action == null ? "[]" : $"[{string.Join(", ", lastInference.action.Select(value => value.ToString("F3")))}]";
                GUILayout.Label($"模型推理：{lastInference.latency_ms:F2} ms｜动作 {action}");
                if (lastInference.blocked)
                    GUILayout.Label($"安全归零：{lastInference.error}");
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
            return policy switch
            {
                "expert" => "专家策略",
                "perturbed" => "受控扰动策略",
                "random" => "随机策略",
                "learned" => "学习策略",
                _ => policy,
            };
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
            if (wristCamera != null)
                Destroy(wristCamera.gameObject);
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
