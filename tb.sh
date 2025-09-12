#!/bin/bash
# TensorBoard启动脚本
# 用法: ./tb.sh [log_directory] [port]
# 示例: ./tb.sh ./workdir/tensorboard 6006

# 设置默认值
DEFAULT_LOGDIR="./workdir/VAP_Xray14_small_t2i_deep/default/tensorboard"
DEFAULT_PORT=6006

# 获取命令行参数
LOGDIR=${1:-$DEFAULT_LOGDIR}
PORT=${2:-$DEFAULT_PORT}

# 检查日志目录是否存在
if [ ! -d "$LOGDIR" ]; then
    echo "警告: 日志目录 '$LOGDIR' 不存在"
    echo "正在创建目录..."
    mkdir -p "$LOGDIR"
fi

echo "启动TensorBoard..."
echo "日志目录: $LOGDIR"
echo "端口: $PORT"
echo "访问地址: http://localhost:$PORT"
echo "按 Ctrl+C 停止服务"
echo ""

# 启动TensorBoard
tensorboard --logdir="$LOGDIR" --port="$PORT" --host=0.0.0.0