pid=$(pgrep -f "tritonserver")

if [ -z "$pid" ]; then
  echo "未找到名为 'tritonserver' 的进程"
else
  # 杀死进程
  kill -9 "$pid"
  echo "已成功杀死进程 $pid"
fi
