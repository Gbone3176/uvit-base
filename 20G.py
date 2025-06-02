import torch
import time



def occupy_gpu_memory(target_gb=20):
    """
    尝试在GPU上分配指定大小的显存。

    Args:
        target_gb (int): 目标分配的显存大小 (GB)。
    """
    if not torch.cuda.is_available():
        print("CUDA 不可用，无法分配 GPU 显存。")
        return

    print(f"当前CUDA设备: {torch.cuda.get_device_name(0)}")
    
    target_bytes = target_gb * 1024 * 1024 * 1024
    allocated_tensors = []
    allocated_bytes = 0

    # 每个元素是 float32 (4 bytes)
    # 每次分配一小块，例如 256MB
    chunk_size_bytes = 256 * 1024 * 1024 
    elements_per_chunk = chunk_size_bytes // 4

    print(f"目标分配显存: {target_gb} GB ({target_bytes / (1024**3):.2f} GB)")
    print(f"每次分配块大小: {chunk_size_bytes / (1024**2):.2f} MB")

    try:
        initial_allocated = torch.cuda.memory_allocated()
        print(f"初始已分配显存: {initial_allocated / (1024**3):.2f} GB")

        while allocated_bytes < target_bytes:
            # 检查剩余需要分配的量，避免超出目标太多
            remaining_bytes_to_target = target_bytes - allocated_bytes
            current_chunk_elements = elements_per_chunk
            
            if remaining_bytes_to_target < chunk_size_bytes:
                current_chunk_elements = remaining_bytes_to_target // 4
                if current_chunk_elements == 0: # 如果剩余太小，不足以分配一个元素，则停止
                    break

            if current_chunk_elements <= 0: # 确保分配正数个元素
                break

            try:
                # 创建一个tensor并移动到GPU
                tensor = torch.randn(current_chunk_elements, device='cuda', dtype=torch.float32)
                allocated_tensors.append(tensor)
                
                # 更新已分配字节数 (torch.cuda.memory_allocated() 更准确)
                current_allocated_total = torch.cuda.memory_allocated()
                allocated_this_iteration = current_allocated_total - initial_allocated - allocated_bytes
                allocated_bytes += allocated_this_iteration # 使用实际增加的分配量

                print(f"已分配 {len(allocated_tensors)} 块, "
                      f"当前总共尝试分配 (脚本内计数): {allocated_bytes / (1024**3):.2f} GB / {target_gb} GB. "
                      f"PyTorch报告总已分配: {current_allocated_total / (1024**3):.2f} GB")

                if allocated_bytes >= target_bytes:
                    print(f"已达到目标分配显存: {allocated_bytes / (1024**3):.2f} GB")
                    break
                
                # 稍微暂停一下，避免过于频繁的打印和分配，给系统一点喘息时间
                # time.sleep(0.01)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"GPU显存不足，停止分配。错误: {e}")
                    current_allocated_total = torch.cuda.memory_allocated()
                    allocated_bytes = current_allocated_total - initial_allocated
                    print(f"最终实际由脚本分配的显存 (脚本内计数): {allocated_bytes / (1024**3):.2f} GB")
                    print(f"PyTorch报告总已分配: {current_allocated_total / (1024**3):.2f} GB")
                    break
                else:
                    raise e # 重新抛出其他运行时错误
            
            if current_chunk_elements < elements_per_chunk and allocated_bytes < target_bytes:
                 # 如果最后一块小于标准块大小且未达到目标，说明可能无法精确达到目标
                 print(f"最后一次分配块较小，可能无法精确达到目标 {target_gb} GB。")


    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        current_allocated_total_final = torch.cuda.memory_allocated()
        script_allocated_final = current_allocated_total_final - initial_allocated
        print(f"\n--- 分配结束 ---")
        print(f"脚本尝试追踪的分配量: {allocated_bytes / (1024**3):.2f} GB")
        print(f"PyTorch 报告最终总已分配显存: {current_allocated_total_final / (1024**3):.2f} GB")
        print(f"其中由本脚本新分配的量 (根据PyTorch报告): {script_allocated_final / (1024**3):.2f} GB")
        print(f"共保留 {len(allocated_tensors)} 个Tensor在显存中。")
        
        if len(allocated_tensors) > 0:
            print("程序将保持运行以占用显存。按 Ctrl+C 退出并释放显存。")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n正在退出并释放显存...")
                del allocated_tensors # 显式删除，帮助垃圾回收
                torch.cuda.empty_cache() # 清理缓存
                print("显存已尝试释放。")
        else:
            print("没有成功分配任何显存。")

if __name__ == "__main__":
    # 你可以在这里修改目标GB数
    occupy_gpu_memory(target_gb=20)