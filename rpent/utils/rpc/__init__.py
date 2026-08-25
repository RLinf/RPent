# RPC 子包 — 进程间通信层
#
# 提供三个模块：
#   rpc        核心 RPC 协议 (RpcClient, RpcFacade, 工具函数)
#   http_rpc   HTTP 传输实现 (HttpRpcServer, HttpRpcClient)
#   socket_rpc TCP/pickle 传输实现 (SocketRpcServer, SocketRpcClient)
#
# 从 rpent.utils.rpc 可直接导入 RpcClient / RpcFacade / wait_for_ready 等，
# 传输层实现需从具体子模块导入。