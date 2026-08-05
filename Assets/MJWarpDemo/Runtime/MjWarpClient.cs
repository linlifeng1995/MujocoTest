using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace MJWarpDemo
{
    public sealed class MjWarpClient : IDisposable
    {
        public const int ProtocolVersion = 1;
        private const int MaxMessageBytes = 64 * 1024 * 1024;

        private readonly SemaphoreSlim requestLock = new SemaphoreSlim(1, 1);
        private TcpClient tcpClient;
        private NetworkStream stream;
        private int nextRequestId;

        public bool IsConnected => tcpClient != null && tcpClient.Connected && stream != null;

        public async Task ConnectAsync(string host, int port, int timeoutMilliseconds, CancellationToken cancellationToken)
        {
            DisposeSocket();
            tcpClient = new TcpClient { NoDelay = true };
            Task connectTask = tcpClient.ConnectAsync(host, port);
            Task timeoutTask = Task.Delay(timeoutMilliseconds, cancellationToken);
            Task completed = await Task.WhenAny(connectTask, timeoutTask);
            if (completed != connectTask)
            {
                DisposeSocket();
                cancellationToken.ThrowIfCancellationRequested();
                throw new TimeoutException($"连接 MJWarp 后端 {host}:{port} 超时");
            }
            await connectTask;
            stream = tcpClient.GetStream();
        }

        public async Task<ResponseEnvelope> SendAsync(string messageType, object payload, CancellationToken cancellationToken)
        {
            if (!IsConnected)
                throw new InvalidOperationException("尚未连接 MJWarp 后端");

            await requestLock.WaitAsync(cancellationToken);
            try
            {
                int requestId = Interlocked.Increment(ref nextRequestId);
                var message = payload == null
                    ? new Dictionary<string, object>()
                    : JsonConvert.DeserializeObject<Dictionary<string, object>>(JsonConvert.SerializeObject(payload));
                message["protocol_version"] = ProtocolVersion;
                message["type"] = messageType;
                message["request_id"] = requestId;

                byte[] json = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(message, Formatting.None));
                if (json.Length > MaxMessageBytes)
                    throw new InvalidOperationException($"协议消息超过大小上限：{MaxMessageBytes} 字节");
                byte[] header = BitConverter.GetBytes(json.Length);
                await stream.WriteAsync(header, 0, header.Length, cancellationToken);
                await stream.WriteAsync(json, 0, json.Length, cancellationToken);
                await stream.FlushAsync(cancellationToken);

                byte[] responseHeader = new byte[4];
                await ReadExactAsync(responseHeader, cancellationToken);
                int responseLength = BitConverter.ToInt32(responseHeader, 0);
                if (responseLength <= 0 || responseLength > MaxMessageBytes)
                    throw new InvalidDataException($"后端响应长度无效：{responseLength}");
                byte[] responseBytes = new byte[responseLength];
                await ReadExactAsync(responseBytes, cancellationToken);
                ResponseEnvelope response = JsonConvert.DeserializeObject<ResponseEnvelope>(Encoding.UTF8.GetString(responseBytes));
                if (response == null)
                    throw new InvalidDataException("后端返回了空 JSON 响应");
                if (response.request_id != requestId)
                    throw new InvalidDataException($"响应请求 ID 不匹配：{response.request_id} != {requestId}");
                if (response.type == "error")
                    throw new InvalidOperationException(response.error == null ? "未知的 MJWarp 后端错误" : $"MJWarp 后端错误：{response.error}");
                return response;
            }
            finally
            {
                requestLock.Release();
            }
        }

        private async Task ReadExactAsync(byte[] buffer, CancellationToken cancellationToken)
        {
            int offset = 0;
            while (offset < buffer.Length)
            {
                int read = await stream.ReadAsync(buffer, offset, buffer.Length - offset, cancellationToken);
                if (read == 0)
                    throw new EndOfStreamException("MJWarp 后端已关闭连接");
                offset += read;
            }
        }

        public void Dispose()
        {
            DisposeSocket();
            requestLock.Dispose();
        }

        private void DisposeSocket()
        {
            stream?.Dispose();
            tcpClient?.Dispose();
            stream = null;
            tcpClient = null;
        }
    }
}
