using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Rendering;

namespace MJWarpDemo
{
    public sealed class MultiModalCapture : IDisposable
    {
        public const int Width = 320;
        public const int Height = 240;

        private readonly Camera sourceCamera;
        private readonly Camera rgbCamera;
        private readonly Camera depthCamera;
        private readonly Camera instanceCamera;
        private readonly Camera wristRgbCamera;
        private readonly Camera wristInstanceCamera;
        private readonly RenderTexture rgbTexture;
        private readonly RenderTexture depthTexture;
        private readonly RenderTexture instanceTexture;
        private readonly RenderTexture wristRgbTexture;
        private readonly RenderTexture wristInstanceTexture;
        private readonly Material depthMaterial;
        private IReadOnlyList<RendererBinding> bindings = Array.Empty<RendererBinding>();
        private Camera wristSourceCamera;

        public RenderTexture RgbTexture => rgbTexture;
        public RenderTexture DepthTexture => depthTexture;
        public RenderTexture InstanceTexture => instanceTexture;

        public MultiModalCapture(Camera sourceCamera)
        {
            this.sourceCamera = sourceCamera;
            rgbCamera = CreateCamera("MJWarp RGB Capture");
            depthCamera = CreateCamera("MJWarp Depth Capture");
            instanceCamera = CreateCamera("MJWarp Instance Capture");
            wristRgbCamera = CreateCamera("MJWarp Wrist RGB Capture");
            wristInstanceCamera = CreateCamera("MJWarp Wrist Instance Capture");

            rgbTexture = CreateTexture("MJWarp RGB", RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB, 24);
            depthTexture = CreateTexture("MJWarp Linear Depth", RenderTextureFormat.RFloat, RenderTextureReadWrite.Linear, 24);
            instanceTexture = CreateTexture("MJWarp Instance", RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear, 24);
            wristRgbTexture = CreateTexture("MJWarp Wrist RGB", RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB, 24);
            wristInstanceTexture = CreateTexture("MJWarp Wrist Instance", RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear, 24);
            rgbCamera.targetTexture = rgbTexture;
            depthCamera.targetTexture = depthTexture;
            instanceCamera.targetTexture = instanceTexture;
            wristRgbCamera.targetTexture = wristRgbTexture;
            wristInstanceCamera.targetTexture = wristInstanceTexture;

            Shader depthShader = Shader.Find("MJWarp/LinearDepth");
            if (depthShader == null)
                throw new InvalidOperationException("缺少 MJWarp/LinearDepth 深度采集 Shader");
            depthMaterial = new Material(depthShader) { name = "MJWarp Linear Depth Material" };
        }

        public void SetBindings(IReadOnlyList<RendererBinding> rendererBindings)
        {
            bindings = rendererBindings ?? Array.Empty<RendererBinding>();
        }

        public void SetWristSource(Camera camera)
        {
            wristSourceCamera = camera;
        }

        public async Task<CapturePayload> CaptureAsync(int frameId)
        {
            SyncCamera(rgbCamera, sourceCamera);
            SyncCamera(depthCamera, sourceCamera);
            SyncCamera(instanceCamera, sourceCamera);
            SyncCamera(wristRgbCamera, wristSourceCamera != null ? wristSourceCamera : sourceCamera);
            SyncCamera(wristInstanceCamera, wristSourceCamera != null ? wristSourceCamera : sourceCamera);

            SetAnnotationsVisible(false);
            try
            {
                rgbCamera.Render();
                wristRgbCamera.Render();
                SwapMaterials(depthMaterial, false);
                depthCamera.Render();
                RestoreMaterials();
                SwapMaterials(null, true);
                instanceCamera.Render();
                wristInstanceCamera.Render();
                RestoreMaterials();
            }
            finally
            {
                RestoreMaterials();
                SetAnnotationsVisible(true);
            }

            Task<byte[]> rgb = ReadbackAsync(rgbTexture, TextureFormat.RGBA32);
            Task<byte[]> depth = ReadbackAsync(depthTexture, TextureFormat.RFloat);
            Task<byte[]> instance = ReadbackAsync(instanceTexture, TextureFormat.RGBA32);
            Task<byte[]> wristRgb = ReadbackAsync(wristRgbTexture, TextureFormat.RGBA32);
            Task<byte[]> wristInstance = ReadbackAsync(wristInstanceTexture, TextureFormat.RGBA32);
            await Task.WhenAll(rgb, depth, instance, wristRgb, wristInstance);
            return new CapturePayload
            {
                FrameId = frameId,
                Rgb = rgb.Result,
                Depth = depth.Result,
                Instance = instance.Result,
                WristRgb = wristRgb.Result,
                WristInstance = wristInstance.Result,
            };
        }

        private void SwapMaterials(Material sharedOverride, bool useInstanceMaterials)
        {
            foreach (RendererBinding binding in bindings)
            {
                if (binding.Renderer == null)
                    continue;
                binding.Renderer.sharedMaterial = useInstanceMaterials ? binding.InstanceMaterial : sharedOverride;
            }
        }

        private void RestoreMaterials()
        {
            foreach (RendererBinding binding in bindings)
            {
                if (binding.Renderer != null)
                    binding.Renderer.sharedMaterial = binding.OriginalMaterial;
            }
        }

        private void SetAnnotationsVisible(bool visible)
        {
            foreach (RendererBinding binding in bindings)
            {
                if (binding.IsAnnotation && binding.Renderer != null)
                    binding.Renderer.enabled = visible;
            }
        }

        private static Task<byte[]> ReadbackAsync(RenderTexture texture, TextureFormat format)
        {
            var completion = new TaskCompletionSource<byte[]>(TaskCreationOptions.RunContinuationsAsynchronously);
            AsyncGPUReadback.Request(texture, 0, format, request =>
            {
                if (request.hasError)
                {
                    completion.TrySetException(new InvalidOperationException($"GPU 回读失败：{texture.name}"));
                    return;
                }
                var native = request.GetData<byte>();
                var bytes = new byte[native.Length];
                native.CopyTo(bytes);
                completion.TrySetResult(bytes);
            });
            return completion.Task;
        }

        private Camera CreateCamera(string name)
        {
            var gameObject = new GameObject(name) { hideFlags = HideFlags.HideAndDontSave };
            Camera camera = gameObject.AddComponent<Camera>();
            camera.enabled = false;
            return camera;
        }

        private static RenderTexture CreateTexture(string name, RenderTextureFormat format, RenderTextureReadWrite readWrite, int depthBits)
        {
            var texture = new RenderTexture(Width, Height, depthBits, format, readWrite)
            {
                name = name,
                antiAliasing = 1,
                useMipMap = false,
                autoGenerateMips = false,
            };
            texture.Create();
            return texture;
        }

        private static void SyncCamera(Camera camera, Camera source)
        {
            RenderTexture target = camera.targetTexture;
            camera.CopyFrom(source);
            camera.enabled = false;
            camera.targetTexture = target;
            camera.transform.SetPositionAndRotation(source.transform.position, source.transform.rotation);
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = Color.black;
        }

        public void Dispose()
        {
            RestoreMaterials();
            if (depthMaterial != null)
                UnityEngine.Object.Destroy(depthMaterial);
            DestroyCamera(rgbCamera);
            DestroyCamera(depthCamera);
            DestroyCamera(instanceCamera);
            DestroyCamera(wristRgbCamera);
            DestroyCamera(wristInstanceCamera);
            DestroyTexture(rgbTexture);
            DestroyTexture(depthTexture);
            DestroyTexture(instanceTexture);
            DestroyTexture(wristRgbTexture);
            DestroyTexture(wristInstanceTexture);
        }

        private static void DestroyCamera(Camera camera)
        {
            if (camera != null)
                UnityEngine.Object.Destroy(camera.gameObject);
        }

        private static void DestroyTexture(RenderTexture texture)
        {
            if (texture == null)
                return;
            texture.Release();
            UnityEngine.Object.Destroy(texture);
        }
    }
}
