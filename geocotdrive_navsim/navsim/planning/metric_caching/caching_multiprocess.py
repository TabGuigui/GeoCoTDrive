from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging
import os
import gc
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from nuplan.planning.training.experiments.cache_metadata_entry import (
    CacheMetadataEntry,
    CacheResult,
    save_cache_metadata,
)

from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from navsim.common.dataloader import SceneLoader, SceneFilter
from navsim.common.dataclasses import SensorConfig, Scene

# 全局logger初始化
logger = logging.getLogger(__name__)


def cache_single_scenario(
    scene_dict: Dict[str, Any], 
    processor: MetricCacheProcessor,
    num_history_frames: int,
    num_future_frames: int
) -> Optional[CacheMetadataEntry]:
    """
    单独处理单个场景的缓存逻辑【核心函数，可直接断点】
    （修改：移除cfg依赖，直接传入必要参数，避免多进程序列化问题）
    :param scene_dict: 单场景的原始数据字典
    :param processor: 缓存处理器
    :param num_history_frames: 历史帧数
    :param num_future_frames: 未来帧数
    :return: 缓存元数据，失败返回None
    """
    try:
        # 构建Scene对象
        scene = Scene.from_scene_dict_list(
            scene_dict,
            None,
            num_history_frames=num_history_frames,
            num_future_frames=num_future_frames,
            sensor_config=SensorConfig.build_no_sensors(),
        )
        # 构建NavSimScenario数据集对象
        scenario = NavSimScenario(
            scene, 
            map_root=os.environ["NUPLAN_MAPS_ROOT"], 
            map_version="nuplan-maps-v1.0"
        )
        # 核心缓存计算逻辑
        return processor.compute_metric_cache(scenario)
    except Exception as e:
        logger.error(f"单场景处理失败: {str(e)}\n{traceback.format_exc()}")
        return None


def process_log_file_task(
    log_file: str,
    tokens: List[str],
    navsim_log_path: Path,
    cache_path: Path,
    force_feature_computation: bool,
    num_history_frames: int,
    num_future_frames: int
) -> Dict[str, Any]:
    """
    多进程任务函数：处理单个日志文件的所有场景
    （独立函数，可被进程池调用，避免共享资源问题）
    """
    # 子进程重新配置logger（确保日志输出到文件）
    local_logger = logging.getLogger(__name__)
    
    # 初始化当前日志文件的过滤器和加载器
    single_log_filter = SceneFilter(
        num_history_frames=num_history_frames,
        num_future_frames=num_future_frames,
        log_names=[log_file],
        tokens=tokens
    )
    single_log_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=navsim_log_path,
        scene_filter=single_log_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    
    scene_frames_dicts = single_log_loader.scene_frames_dicts
    scene_count = len(scene_frames_dicts)
    local_logger.info(f"\n===== 子进程处理日志文件: {log_file} (共{scene_count}个场景) =====")

    # 子进程独立创建处理器（避免多进程共享文件句柄）
    processor = MetricCacheProcessor(
        cache_path=cache_path,
        force_feature_computation=force_feature_computation,
    )

    # 统计当前日志文件的处理结果
    success_count = 0
    failure_count = 0
    metadata_list = []

    # 串行处理当前日志文件的所有场景（单日志内仍串行，跨日志并行）
    for scene_idx, scene_dict in enumerate(scene_frames_dicts.values()):
        scene_idx_str = f"{scene_idx+1}/{scene_count}"
        local_logger.info(f"处理场景: {scene_idx_str} | log_file={log_file}")
        
        # 处理单个场景
        cache_meta = cache_single_scenario(
            scene_dict=scene_dict,
            processor=processor,
            num_history_frames=num_history_frames,
            num_future_frames=num_future_frames
        )
        
        # 更新统计
        if cache_meta is not None:
            success_count += 1
            metadata_list.append(cache_meta)
        else:
            failure_count += 1
            local_logger.warning(f"场景 {scene_idx_str} | log_file={log_file} 缓存失败")
        
        # 子进程内强制GC，防止内存泄漏
        gc.collect()

    local_logger.info(f"日志文件 {log_file} 处理完成 | 成功: {success_count} | 失败: {failure_count}")
    return {
        "log_file": log_file,
        "success": success_count,
        "failure": failure_count,
        "metadata": metadata_list
    }


def cache_data(cfg: DictConfig) -> None:
    """
    【多进程主入口】并行处理日志文件，日志输出到cache_info.txt
    :param cfg: omegaconf 全局配置字典
    """
    # ======================
    # 1. 配置日志：同时输出到文件和控制台
    # ======================
    txt_path = os.path.join(
        cfg.cache.cache_path, cfg.train_test_split.data_split + "_cache_info.txt"
    )
    log_file = Path(txt_path)
    # 清除原有handler，避免重复输出
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 文件handler（保存到cache_info.txt，UTF-8编码避免乱码）
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # 控制台handler（保留终端输出）
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # 添加handler并设置日志级别
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    # ======================
    # 2. 基础校验和路径处理（保留原有逻辑）
    # ======================
    assert cfg.cache.cache_path is not None, f"Cache path cannot be None when caching, got {cfg.cache.cache_path}"
    cache_path = Path(cfg.cache.cache_path)
    cfg.navsim_log_path = cfg.navsim_log_path.replace('navsim_logs', 'meta_datas') 
    navsim_log_path = Path(cfg.navsim_log_path)
    
    # ======================
    # 3. 加载待处理的日志文件列表
    # ======================
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=navsim_log_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    # 构建多进程任务列表
    task_list = []
    log_tokens_map = scene_loader.get_tokens_list_per_log()
    for log_file, tokens_list in log_tokens_map.items():
        task_list.append({
            "log_file": log_file,
            "tokens": tokens_list,
            "navsim_log_path": navsim_log_path,
            "cache_path": cache_path,
            "force_feature_computation": cfg.cache.force_feature_computation,
            "num_history_frames": cfg.train_test_split.scene_filter.num_history_frames,
            "num_future_frames": cfg.train_test_split.scene_filter.num_future_frames
        })
    
    logger.info(f"===== 多进程缓存启动 =====")
    logger.info(f"待处理日志文件数: {len(task_list)}")
    logger.info(f"CPU核心数: {cpu_count()} | 进程池大小: {min(cpu_count(), len(task_list))}")

    # ======================
    # 4. 多进程执行任务
    # ======================
    total_successes = 0
    total_failures = 0
    all_cache_metadata: List[CacheMetadataEntry] = []

    # 创建进程池（max_workers建议不超过CPU核心数，避免资源竞争）
    max_workers = getattr(cfg, "num_workers", cpu_count())
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_map = {
            executor.submit(
                process_log_file_task,
                task["log_file"],
                task["tokens"],
                task["navsim_log_path"],
                task["cache_path"],
                task["force_feature_computation"],
                task["num_history_frames"],
                task["num_future_frames"]
            ): task for task in task_list
        }

        # 遍历完成的任务，汇总结果
        for future in as_completed(future_map):
            task = future_map[future]
            log_file = task["log_file"]
            try:
                # 获取任务结果
                result = future.result()
                total_successes += result["success"]
                total_failures += result["failure"]
                all_cache_metadata.extend(result["metadata"])
                logger.info(f"汇总 | 日志文件 {log_file} | 成功: {result['success']} | 失败: {result['failure']}")
            except Exception as e:
                # 捕获进程执行异常，标记该日志文件所有场景失败
                total_failures += len(task["tokens"])
                error_msg = f"日志文件 {log_file} 执行异常: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)

    # ======================
    # 5. 结果统计和元数据保存
    # ======================
    total_scenarios = total_successes + total_failures
    logger.info(f"\n===== 多进程缓存完成 =====")
    if total_failures == 0:
        logger.info(f"✅ 全部成功 | 总计场景数: {total_scenarios}")
    else:
        logger.info(f"⚠️ 部分失败 | 成功: {total_successes} | 失败: {total_failures} | 总计: {total_scenarios}")

    # 保存元数据（保留原有逻辑）
    node_id = 0
    logger.info(f"保存缓存元数据到: {cache_path}")
    save_cache_metadata(all_cache_metadata, cache_path, node_id)
    logger.info("✅ 元数据文件保存完成！")

    # 关闭日志handler
    file_handler.close()
    logger.removeHandler(file_handler)
    logger.removeHandler(console_handler)


# 测试入口（可选）
if __name__ == "__main__":
    # 注意：多进程必须在if __name__ == "__main__"中执行，否则会报错
    import hydra
    @hydra.main(config_path="configs", config_name="cache_config")
    def main(cfg: DictConfig):
        cache_data(cfg)
    
    main()