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
        private readonly RenderTexture rgbTexture;
        private readonly RenderTexture depthTexture;
        private readonly RenderTexture instanceTexture;
        private readonly Material depthMaterial;
        private IReadOnlyList<RendererBinding> bindings = Array.Empty<RendererBinding>();

        public RenderTexture RgbTexture => rgbTexture;
        public RenderTexture DepthTexture => depthTexture;
        public RenderTexture InstanceTexture => instanceTexture;

        public MultiModalCapture(Camera sourceCamera)
        {
            this.sourceCamera = sourceCamera;
            rgbCamera = CreateCamera("MJWarp RGB Capture");
            depthCamera = CreateCamera("MJWarp Depth Capture");
            instanceCamera = CreateCamera("MJWarp Instance Capture");

            rgbTexture = CreateTexture("MJWarp RGB", RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB, 24);
            depthTexture = CreateTexture("MJWarp Linear Depth", RenderTextureFormat.RFloat, RenderTextureReadWrite.Linear, 24);
            instanceTexture = CreateTexture("MJWarp Instance", RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear, 24);
            rgbCamera.targetTexture = rgbTexture;
            depthCamera.targetTexture = depthTexture;
            instanceCamera.targetTexture = instanceTexture;

            Shader depthShader = Shader.Find("MJWarp/LinearDepth");
            if (depthShader == null)
                throw new InvalidOperationException("Missing shader MJWarp/LinearDepth");
            depthMaterial = new Material(depthShader) { name = "MJWarp Linear Depth Material" };
        }

        public void SetBindings(IReadOnlyList<RendererBinding> rendererBindings)
        {
            bindings = rendererBindings ?? Array.Empty<RendererBinding>();
        }

        public async Task<CapturePayload> CaptureAsync(int frameId)
        {
            SyncCamera(rgbCamera);
            SyncCamera(depthCamera);
            SyncCamera(instanceCamera);

            rgbCamera.Render();
            SwapMaterials(depthMaterial, false);
            depthCamera.Render();
            RestoreMaterials();
            SwapMaterials(null, true);
            instanceCamera.Render();
            RestoreMaterials();

            Task<byte[]> rgb = ReadbackAsync(rgbTexture, TextureFormat.RGBA32);
            Task<byte[]> depth = ReadbackAsync(depthTexture, TextureFormat.RFloat);
            Task<byte[]> instance = ReadbackAsync(instanceTexture, TextureFormat.RGBA32);
            await Task.WhenAll(rgb, depth, instance);
            return new CapturePayload
            {
                FrameId = frameId,
                Rgb = rgb.Result,
                Depth = depth.Result,
                Instance = instance.Result,
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

        private static Task<byte[]> ReadbackAsync(RenderTexture texture, TextureFormat format)
        {
            var completion = new TaskCompletionSource<byte[]>(TaskCreationOptions.RunContinuationsAsynchronously);
            AsyncGPUReadback.Request(texture, 0, format, request =>
            {
                if (request.hasError)
                {
                    completion.TrySetException(new InvalidOperationException($"GPU readback failed for {texture.name}"));
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

        private void SyncCamera(Camera camera)
        {
            RenderTexture target = camera.targetTexture;
            camera.CopyFrom(sourceCamera);
            camera.enabled = false;
            camera.targetTexture = target;
            camera.transform.SetPositionAndRotation(sourceCamera.transform.position, sourceCamera.transform.rotation);
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
            DestroyTexture(rgbTexture);
            DestroyTexture(depthTexture);
            DestroyTexture(instanceTexture);
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
